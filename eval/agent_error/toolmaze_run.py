"""
在运行时给 ToolMaze 打补丁(不修改上游源码),两种用法:

A. 命令行:像官方 run_eval.py 一样跑(E1-API 试跑用这种)
    python eval/agent_error/toolmaze_run.py --config eval/agent_error/configs/toolmaze_api_pilot.yaml --task-category c1 --limit 5
   其余参数原样传给 run_eval.py。
B. 导入:run_e2.py 调用 apply_patches(cfg, cfg_path),再逐回合调用上游的 evaluate_single_task。

补丁(都打在 evaluation/agents/openai_agent.py 的 OpenAIAgent 上;VLLMAgent 继承它,本地模型同样生效):
1. 单次请求超时:官方写死 timeout=45.0,改为 patch.request_timeout_s。
2. extra_body:agent.extra_body(如 {"enable_thinking": false})加进每次请求。
3. 调用日志:每次请求(含重试)写一行 JSON:token、思考 token、缓存命中、延迟、等待限速的秒数、finish_reason、错误,
   以及 run_e2 设置的回合标签。
4. 限速(patch.rate_limit,可选):按 token 和请求数的令牌桶(ratelimit.py),每次请求前估算 token,额度不够就等。
5. 重试(patch.retry,可选):429 / 超时 / 连接错误 / 5xx 指数退避加抖动;OpenAI 客户端自带的重试关掉
   (max_retries=0),由这里统一处理并计数。重试仍失败时计入本回合的 api_failures 再抛出——上游会把异常
   变成 "Error during execution" 最终回答,run_e2 据 api_failures 判定整对重跑。
6. Outcome Monitor(E2):包装 receive_tool_result。只有当 run_e2 在当前线程设置了 EPISODE.monitor 时才生效;
   用 receipt.augment 检查工具结果,违反时把 {"tool_result", "outcome_contract"} 交给模型。
   判分读的是 ExecutionEngine 记录的原始结果(log_round 在 receive_tool_result 之前),不受影响。
"""
from __future__ import annotations

import json
import os
import random
import runpy
import sys
import threading
import time
from typing import Any, Callable

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TM = os.environ.get("TOOLMAZE_DIR", os.path.join(ROOT, "external", "agent_error", "toolmaze"))
sys.path.insert(0, HERE)
from ratelimit import Limits, TokenBucket, TokenEstimator  # noqa: E402
from receipt import augment  # noqa: E402

# run_e2 在每个工作线程里设置:labels(dict)、monitor(None 或 {registry, subs, variant, log})、stats(dict)
EPISODE = threading.local()
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_log_lock = threading.Lock()


def new_stats() -> dict:
    return {"calls": 0, "attempts": 0, "retries": 0, "api_failures": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "max_prompt_tokens": 0, "length_finishes": 0, "wait_s": 0.0, "latency_s": 0.0, "receipts": 0}


def _stats() -> dict | None:
    return getattr(EPISODE, "stats", None)


def append_jsonl(path: str | None, rec: dict) -> None:
    if not path:
        return
    with _log_lock, open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}:
        return True
    return getattr(exc, "status_code", None) in RETRY_STATUS


def call_with_retry(fn: Callable[[], Any], max_attempts: int, base_s: float, cap_s: float,
                    sleep: Callable[[float], None] = time.sleep, rng: random.Random | None = None,
                    on_retry: Callable[[BaseException, float], None] | None = None) -> Any:
    """暂时性错误指数退避(base·2^i,封顶 cap,乘 0.5–1.0 抖动);非暂时性错误或次数用完直接抛出。"""
    rng = rng or random.Random()
    for i in range(max_attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not is_transient(e) or i == max_attempts - 1:
                raise
            delay = min(cap_s, base_s * 2 ** i) * (0.5 + rng.random() / 2)
            if on_retry:
                on_retry(e, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def make_create_wrapper(create: Callable, *, timeout: float, extra_body: dict | None, log_path: str | None,
                        limiter: TokenBucket | None, estimator: TokenEstimator | None, retry: dict | None) -> Callable:
    def wrapped(*args, **kw):
        kw["timeout"] = timeout
        if extra_body:
            kw["extra_body"] = {**(kw.get("extra_body") or {}), **extra_body}
        chars = len(json.dumps(kw.get("messages") or [], ensure_ascii=False)) + \
            len(json.dumps(kw.get("tools") or [], ensure_ascii=False))
        est = estimator.estimate(chars, kw.get("max_tokens") or 0) if (limiter and estimator) else 0
        st = _stats()
        labels = dict(getattr(EPISODE, "labels", {}) or {})

        def once():
            waited = limiter.acquire(est) if limiter else 0.0
            t0 = time.time()
            rec = {"ts": round(t0, 3), **labels, "model": kw.get("model"), "n_messages": len(kw.get("messages") or []),
                   "chars": chars, "est_tokens": est, "wait_s": round(waited, 2)}
            if st is not None:
                st["attempts"] += 1
                st["wait_s"] += waited
            try:
                r = create(*args, **kw)
            except Exception as e:
                if limiter:
                    limiter.settle(est, 0)          # 被拒的请求不消耗 token
                rec.update(ok=False, latency_s=round(time.time() - t0, 2), error=f"{type(e).__name__}: {str(e)[:300]}")
                append_jsonl(log_path, rec)
                raise
            u = getattr(r, "usage", None)
            pt, ct = getattr(u, "prompt_tokens", None) or 0, getattr(u, "completion_tokens", None) or 0
            ctd, ptd = getattr(u, "completion_tokens_details", None), getattr(u, "prompt_tokens_details", None)
            ch = r.choices[0] if getattr(r, "choices", None) else None
            msg = getattr(ch, "message", None)
            if limiter:
                limiter.settle(est, pt + ct)
            if estimator:
                estimator.observe(chars, pt)
            lat = round(time.time() - t0, 2)
            rec.update(ok=True, latency_s=lat, prompt_tokens=pt, completion_tokens=ct,
                       reasoning_tokens=getattr(ctd, "reasoning_tokens", None),
                       cached_tokens=getattr(ptd, "cached_tokens", None),
                       has_reasoning_content=bool(getattr(msg, "reasoning_content", None)),
                       finish_reason=getattr(ch, "finish_reason", None))
            append_jsonl(log_path, rec)
            if st is not None:
                st["calls"] += 1
                st["prompt_tokens"] += pt
                st["completion_tokens"] += ct
                st["max_prompt_tokens"] = max(st["max_prompt_tokens"], pt)
                st["length_finishes"] += rec["finish_reason"] == "length"
                st["latency_s"] += lat
            return r

        if not retry:
            return once()

        def on_retry(e, delay):
            if st is not None:
                st["retries"] += 1

        try:
            return call_with_retry(once, retry.get("max_attempts", 8), retry.get("base_s", 5.0),
                                   retry.get("cap_s", 120.0), on_retry=on_retry)
        except Exception:
            if st is not None:
                st["api_failures"] += 1
            raise

    return wrapped


def call_arguments(messages: list[dict], index: int = 0) -> dict:
    """当前工具调用的参数:最近一条带 tool_calls 的 assistant 消息里第 index 个(越界取第 0 个)。"""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls = msg["tool_calls"]
            tc = calls[index] if index < len(calls) else calls[0]
            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            if isinstance(raw, dict):
                return raw
            try:
                parsed = json.loads(raw or "{}")
            except (TypeError, ValueError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def visible_tools(agent: Any) -> list[str]:
    return [t.get("function", {}).get("name") for t in getattr(agent, "tools", []) or [] if isinstance(t, dict)]


def monitored_receive(orig: Callable) -> Callable:
    def receive(self, tool_name, result, tool_call_index=0):
        mon = getattr(EPISODE, "monitor", None)
        if not mon:
            return orig(self, tool_name, result, tool_call_index)
        args = call_arguments(self.messages, tool_call_index)
        out = augment(tool_name, args, result, mon["registry"].get(tool_name), mon["subs"], visible_tools(self),
                      mon["variant"])
        if out.receipt is not None:
            st = _stats()
            if st is not None:
                st["receipts"] += 1
            append_jsonl(mon.get("log"), {**(getattr(EPISODE, "labels", {}) or {}), "tool": tool_name,
                                          "violations": [v.to_dict() for v in out.violations],
                                          "receipt": out.receipt})
        return orig(self, tool_name, out.model_view, tool_call_index)

    receive.__wrapped_by_outcome_monitor__ = True
    return receive


def _abs(cfg_path: str, p: str | None) -> str | None:
    if not p:
        return None
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(os.path.dirname(cfg_path), p))


def apply_patches(cfg: dict, cfg_path: str, agent_cls: type | None = None) -> dict:
    """给 OpenAIAgent(或测试传入的类)打补丁。返回共享对象,供调用方查看限速状态。只打一次。"""
    if agent_cls is None:
        sys.path.insert(0, TM)
        from evaluation.agents import openai_agent as oa  # noqa: E402
        agent_cls = oa.OpenAIAgent
    if getattr(agent_cls, "__toolmaze_run_patched__", False):
        return agent_cls.__toolmaze_run_shared__
    patch = cfg.get("patch", {}) or {}
    rl = patch.get("rate_limit")
    shared = {
        "limiter": TokenBucket(Limits(rl.get("tpm", 40_000), rl.get("rpm", 1_000), rl.get("headroom", 0.9))) if rl else None,
        "estimator": TokenEstimator(rl.get("chars_per_token", 2.5), rl.get("output_ratio", 0.05)) if rl else None,
        "log": _abs(cfg_path, patch.get("call_log")),
    }
    if shared["log"]:
        os.makedirs(os.path.dirname(shared["log"]), exist_ok=True)
    timeout = float(patch.get("request_timeout_s", 45))
    extra = (cfg.get("agent") or {}).get("extra_body") or None
    retry = patch.get("retry")
    orig_init = agent_cls.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        if retry and hasattr(self.client, "with_options"):
            self.client = self.client.with_options(max_retries=0)
        self.client.chat.completions.create = make_create_wrapper(
            self.client.chat.completions.create, timeout=timeout, extra_body=extra, log_path=shared["log"],
            limiter=shared["limiter"], estimator=shared["estimator"], retry=retry)

    agent_cls.__init__ = init
    agent_cls.receive_tool_result = monitored_receive(agent_cls.receive_tool_result)
    agent_cls.__toolmaze_run_patched__ = True
    agent_cls.__toolmaze_run_shared__ = shared
    print(f"[toolmaze_run] timeout={timeout}s extra_body={extra} retry={retry} rate_limit={rl} call_log={shared['log']}")
    return shared


def _arg(name: str, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    cfg_path = os.path.abspath(_arg("--config"))
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    apply_patches(cfg, cfg_path)
    os.chdir(TM)
    sys.argv = ["run_eval.py"] + sys.argv[1:]
    sys.argv[sys.argv.index("--config") + 1] = cfg_path
    runpy.run_path(os.path.join("evaluation", "scripts", "run_eval.py"), run_name="__main__")


if __name__ == "__main__":
    main()
