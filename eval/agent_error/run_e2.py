"""
E2 编排:按配对随机交错运行基线组与回执组,可中断续跑,API 出错的配对整对重跑并计数。

    python eval/agent_error/run_e2.py --config eval/agent_error/configs/e2_api.yaml --run pilot
    python eval/agent_error/run_e2.py --config eval/agent_error/configs/e2_api.yaml --run pilot --estimate main

- 回合清单来自冻结的 configs/e2_manifest.json:main = C3/C4 × P3/P4 的 80 个工作流;main_ext = main + ② 的补充清单 120(configs/e2_manifest_ext.json);p0 = 同 80 题的 P0;
  c1 = C1 补充;pilot = 清单之外的 C3/C4 P3/P4 题(按层级与模式轮流抽,种子固定),不与正式实验重叠。
- 顺序:配对顺序与每对内两组的先后都由种子打乱,写入 order.json;workers 个配对并行。
- 组(arm):off = 基线(不检查);full = 完整回执;stripped = 去掉备用工具列表的回执。
  第 k 折的题只用注册表 C_k(contracts.registry_for)。
- 记录:每个回合一行写入 episodes.jsonl(token、调用数、重试、api_failures、回执数、是否通过、耗时)。
  配对的状态由最近一次尝试的记录决定:两组都有记录且都没有 api_failures → 完成;否则删掉两组的结果文件整对重跑,
  最多 max_pair_attempts 次,重跑次数写入 summary.json。
- 估算(--estimate RUN):用本次运行(通常是 pilot)的回合记录,外推 RUN 的总 token、耗时与花费,写入日志和 estimate.json。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import contracts as C  # noqa: E402
import e2_manifest as M  # noqa: E402
import receipt as R  # noqa: E402
import toolmaze_run as T  # noqa: E402

ARM_VARIANTS = {"off": None, "full": "full", "stripped": "stripped"}


def arm_variant(value) -> str | None:
    """YAML 会把未加引号的 off 解析成 False,两者都表示基线组。"""
    if value is False:
        return None
    if value not in ARM_VARIANTS:
        raise ValueError(f"未知的组设置 {value!r},只能是 {list(ARM_VARIANTS)}")
    return ARM_VARIANTS[value]


@dataclass(frozen=True)
class Pair:
    tier: str
    task_id: str
    mode: str

    @property
    def key(self) -> str:
        return f"{self.task_id}_{self.mode}"


# ---------------------------------------------------------------- 配置与回合
def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), text)
    return yaml.safe_load(text)


def pilot_pairs(manifest: dict, available: dict[str, list[str]], n: int, seed: int) -> list[Pair]:
    """清单之外的 C3/C4 题,按(c3,P3)(c3,P4)(c4,P3)(c4,P4)轮流各抽一题,共 n 题,任务互不相同。"""
    used = {e["task_id"] for e in manifest["main"]}
    rng = random.Random(seed)
    pools = {t: [x for x in sorted(available[t]) if x not in used] for t in M.MAIN_TIERS}
    for t in pools:
        rng.shuffle(pools[t])
    strata = [(t, m) for t in M.MAIN_TIERS for m in M.MODES]
    out = []
    for i in range(n):
        tier, mode = strata[i % len(strata)]
        out.append(Pair(tier, pools[tier].pop(), mode))
    return out


def pairs_for(group: str, manifest: dict, available: dict[str, list[str]] | None = None, n: int = 5,
              seed: int = 0) -> list[Pair]:
    if group == "main":
        return [Pair(e["tier"], e["task_id"], e["mode"]) for e in manifest["main"]]
    if group == "main_ext":                          # ② 200 对:主清单 80 + 补充清单 120
        return [Pair(e["tier"], e["task_id"], e["mode"]) for e in manifest["main"] + manifest[M.EXT_KEY]]
    if group == "ext":                               # 只取补充清单 120 对(与 main 合起来 = main_ext)
        return [Pair(e["tier"], e["task_id"], e["mode"]) for e in manifest[M.EXT_KEY]]
    if group == "p0":
        return [Pair(e["tier"], e["task_id"], "P0") for e in manifest["main"]]
    if group == "c1":
        return [Pair(e["tier"], e["task_id"], e["mode"]) for e in manifest["c1_supplement"]]
    if group == "pilot":
        return pilot_pairs(manifest, available or {}, n, seed)
    raise ValueError(f"未知回合组 {group!r}")


def plan_order(pairs: list[Pair], arms: list[str], seed: int) -> list[tuple[Pair, list[str]]]:
    rng = random.Random(seed)
    order = list(pairs)
    rng.shuffle(order)
    plan = []
    for p in order:
        a = list(arms)
        rng.shuffle(a)
        plan.append((p, a))
    return plan


# ---------------------------------------------------------------- 续跑状态
def read_records(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pair_state(records: list[dict], pair_key: str, arms: list[str]) -> tuple[str, int]:
    """返回(状态, 最近一次尝试编号)。状态:todo 未跑过;done 最近一次两组都完成且无 API 错误;retry 其他情况。"""
    mine = [r for r in records if r["pair"] == pair_key]
    if not mine:
        return "todo", 0
    last = max(r["attempt"] for r in mine)
    latest = {r["arm"]: r for r in mine if r["attempt"] == last}
    if set(arms) <= set(latest) and all(latest[a]["api_failures"] == 0 for a in arms):
        return "done", last
    return "retry", last


def pair_reruns(records: list[dict]) -> int:
    """整对重跑的总次数 = 各配对的(最大尝试编号 − 1)之和。"""
    last: dict[str, int] = {}
    for r in records:
        last[r["pair"]] = max(last.get(r["pair"], 0), r["attempt"])
    return sum(v - 1 for v in last.values())


# ---------------------------------------------------------------- 执行
class Runner:
    def __init__(self, cfg: dict, cfg_path: str, run: str):
        self.cfg, self.run_name = cfg, run
        e2 = cfg["e2"]
        spec = e2["runs"][run]
        self.arms: dict[str, str | None] = {a: arm_variant(v) for a, v in spec["arms"].items()}
        self.out = T._abs(cfg_path, os.path.join(e2["results_dir"], run))
        os.makedirs(self.out, exist_ok=True)
        cfg.setdefault("patch", {})["call_log"] = os.path.join(self.out, "calls.jsonl")
        self.shared = T.apply_patches(cfg, os.path.join(self.out, "_"))
        self.records_path = os.path.join(self.out, "episodes.jsonl")
        self.receipt_log = os.path.join(self.out, "receipts.jsonl")
        self.max_attempts = e2.get("max_pair_attempts", 3)
        self.workers = e2.get("workers", 1)
        self.regs, self.folds = C.load(os.path.join(C.OUT, e2.get("registry", "gold")))
        self.subs = R.build_subs(os.path.join(T.TM, "tools", "definitions"))
        manifest = M.load()
        available = {t: M.tasks_with_modes(T.TM, t) for t in M.MAIN_TIERS}
        self.pairs = pairs_for(spec["episodes"], manifest, available, spec.get("n", 5), e2["seed"] + 1)
        self.plan = plan_order(self.pairs, list(self.arms), e2["seed"])
        with open(os.path.join(self.out, "order.json"), "w", encoding="utf-8") as fh:
            json.dump({"seed": e2["seed"], **M.hashes(manifest),
                       "bare_object_additional_properties": bool(cfg["patch"].get("bare_object_additional_properties")),
                       "order": [{**asdict(p), "arms": a} for p, a in self.plan]}, fh, indent=1)
        self.lock = threading.Lock()
        self.reruns = 0
        self._upstream()

    def _upstream(self) -> None:
        sys.path.insert(0, T.TM)
        os.chdir(T.TM)
        from evaluation.core.judge import JudgeSystem  # noqa: E402
        from evaluation.scripts import run_eval  # noqa: E402
        from evaluation.utils.result_saver import ResultSaver  # noqa: E402
        self.RE, self.ResultSaver = run_eval, ResultSaver
        self.judge = JudgeSystem(llm_client=None, model="rule-based", template_dir=self.cfg["data"]["template_dir"])
        self.savers: dict[tuple[str, str], object] = {}

    def saver(self, arm: str, tier: str):
        with self.lock:
            if (arm, tier) not in self.savers:
                self.savers[(arm, tier)] = self.ResultSaver(base_dir=os.path.join(self.out, arm),
                                                            model=self.cfg["agent"]["model"], agent_type="fc",
                                                            task_category=tier)
            return self.savers[(arm, tier)]

    def clear(self, p: Pair) -> None:
        for arm in self.arms:
            s = self.saver(arm, p.tier)
            for path in (s.inferences_dir / f"{p.key}_inference.json", s.evaluations_dir / f"{p.key}_eval.json"):
                if path.exists():
                    path.unlink()

    def episode(self, p: Pair, arm: str, attempt: int) -> dict:
        with open(os.path.join("data", "perturbed_tasks", p.tier, f"{p.key}.json"), encoding="utf-8") as fh:
            task = json.load(fh)
        variant = self.arms[arm]
        T.EPISODE.labels = {"run": self.run_name, "pair": p.key, "arm": arm, "attempt": attempt}
        T.EPISODE.stats = T.new_stats()
        T.EPISODE.monitor = None if variant is None else {
            "registry": C.registry_for(self.regs, self.folds, p.task_id), "subs": self.subs,
            "variant": variant, "log": self.receipt_log}
        t0 = time.time()
        try:
            res = self.RE.evaluate_single_task(task, self.cfg, self.cfg["data"]["tools_dir"], self.judge,
                                               self.saver(arm, p.tier))
        finally:
            st, T.EPISODE.monitor = T.EPISODE.stats, None
        inf = self.saver(arm, p.tier).inferences_dir / f"{p.key}_inference.json"
        err_answer = inf.exists() and "Error during execution" in inf.read_text(encoding="utf-8")
        rec = {"pair": p.key, "tier": p.tier, "task_id": p.task_id, "mode": p.mode, "arm": arm, "attempt": attempt,
               "passed": res.get("passed"), "harness_error": res.get("error"), "error_in_answer": err_answer,
               "seconds": round(time.time() - t0, 1), **st}
        T.append_jsonl(self.records_path, rec)
        return rec

    def run_pair(self, p: Pair, arms: list[str]) -> None:
        state, last = pair_state(read_records(self.records_path), p.key, list(self.arms))
        while state != "done":
            if last >= self.max_attempts:
                print(f"[run_e2] {p.key} 已尝试 {last} 次仍有 API 错误,放弃")
                return
            if state == "retry":
                with self.lock:
                    self.reruns += 1
                print(f"[run_e2] {p.key} 第 {last} 次尝试有 API 错误或未完成,整对重跑")
            self.clear(p)
            attempt = last + 1
            recs = [self.episode(p, arm, attempt) for arm in arms]
            state = "done" if all(r["api_failures"] == 0 for r in recs) else "retry"
            last = attempt

    def run(self) -> dict:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(lambda pa: self.run_pair(*pa), self.plan))
        recs = read_records(self.records_path)
        states = [pair_state(recs, p.key, list(self.arms))[0] for p in self.pairs]
        summary = {"run": self.run_name, "pairs": len(self.pairs), "done": states.count("done"),
                   "unresolved": len(states) - states.count("done"), "pair_reruns_this_session": self.reruns,
                   "pair_reruns_total": pair_reruns(recs),
                   "minutes_this_session": round((time.time() - t0) / 60, 1),
                   "limiter_wait_s": round(self.shared["limiter"].waited_s, 1) if self.shared["limiter"] else 0}
        with open(os.path.join(self.out, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1, ensure_ascii=False)
        print(json.dumps(summary, ensure_ascii=False))
        return summary


# ---------------------------------------------------------------- 估算
def estimate(records: list[dict], n_pairs: int, arms: list[str], workers: int, limits: dict | None,
             price: dict | None, num_ctx: int | None = None, max_tokens: int | None = None) -> dict:
    """用试跑回合(每组取最近一次尝试)外推:总 token、耗时下界(限速与延迟两者取大)、花费。"""
    by_arm = {a: [r for r in records if r["arm"] == a] for a in arms}
    per = {}
    for a, rs in by_arm.items():
        if not rs:
            raise ValueError(f"试跑没有 {a} 组的记录")
        n = len(rs)
        per[a] = {"episodes": n, "prompt_tokens": sum(r["prompt_tokens"] for r in rs) / n,
                  "completion_tokens": sum(r["completion_tokens"] for r in rs) / n,
                  "calls": sum(r["calls"] for r in rs) / n, "latency_s": sum(r["latency_s"] for r in rs) / n,
                  "max_prompt_tokens": max(r["max_prompt_tokens"] for r in rs),
                  "receipts": sum(r["receipts"] for r in rs) / n}
    tin = n_pairs * sum(v["prompt_tokens"] for v in per.values())
    tout = n_pairs * sum(v["completion_tokens"] for v in per.values())
    latency_min = n_pairs * sum(v["latency_s"] for v in per.values()) / max(1, workers) / 60
    calls = n_pairs * sum(v["calls"] for v in per.values())
    out = {"n_pairs": n_pairs, "arms": arms, "per_episode": per, "total_prompt_tokens": round(tin),
           "total_completion_tokens": round(tout), "total_calls": round(calls),
           "latency_bound_minutes": round(latency_min, 1)}
    if limits:
        head = limits.get("headroom", 0.9)
        out["rate_limit_bound_minutes"] = round(max((tin + tout) / (limits["tpm"] * head),
                                                    calls / (limits["rpm"] * head)), 1)
    out["estimated_minutes"] = max(out["latency_bound_minutes"], out.get("rate_limit_bound_minutes", 0))
    if price:
        out["estimated_cost_usd"] = round(tin / 1e6 * price["input"] + tout / 1e6 * price["output"], 2)
    if num_ctx and max_tokens:
        out["prompt_budget"] = num_ctx - max_tokens
        out["max_prompt_tokens_seen"] = max(v["max_prompt_tokens"] for v in per.values())
        out["fits_context"] = out["max_prompt_tokens_seen"] < out["prompt_budget"]
    return out


def latest_records(records: list[dict]) -> list[dict]:
    last: dict[tuple, dict] = {}
    for r in records:
        k = (r["pair"], r["arm"])
        if k not in last or r["attempt"] >= last[k]["attempt"]:
            last[k] = r
    return list(last.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--estimate", default=None, help="用本次运行的记录外推另一组运行(如 main)")
    a = ap.parse_args()
    cfg_path = os.path.abspath(a.config)
    cfg = load_cfg(cfg_path)
    runner = Runner(cfg, cfg_path, a.run)
    runner.run()
    if a.estimate:
        e2 = cfg["e2"]
        target = e2["runs"][a.estimate]
        n_pairs = len(pairs_for(target["episodes"], M.load(), n=target.get("n", 5)))
        est = estimate(latest_records(read_records(runner.records_path)), n_pairs, list(runner.arms),
                       e2.get("workers", 1), (cfg.get("patch") or {}).get("rate_limit"), e2.get("price_usd_per_m"),
                       e2.get("num_ctx"), cfg["agent"].get("max_tokens"))
        est["target_run"] = a.estimate
        with open(os.path.join(runner.out, f"estimate_{a.estimate}.json"), "w", encoding="utf-8") as fh:
            json.dump(est, fh, indent=1, ensure_ascii=False)
        print("[run_e2] 估算:" + json.dumps(est, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
