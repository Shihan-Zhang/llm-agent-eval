"""最小复现:API 把无属性对象参数传成 {},是 schema 的问题还是模型的问题?

从 ① API 基线组里按固定规则选一个请求(见 pick_episode),原样重建:上游 ExecutionEngine 给出工具定义,
上游 OpenAIAgent 给出系统提示,推理日志给出用户提示和之前的调用与结果。只改工具 schema 或模型,发 4 个变体:
  A 原 schema(Qwen3.5-35B-A3B)
  B 无属性对象加 additionalProperties: true
  C 按 P0 参考参数补 properties(只做诊断,不打算采用)
  D 原 schema,换 DeepSeek-V3.2
每个变体 3 次,温度 0,请求参数与 ① 相同(parallel_tool_calls=False、max_tokens 16000、enable_thinking=false)。
原始响应里的 arguments 字符串原样存进 results/calls.jsonl;各变体的完整请求(不含密钥)存进 results/request_*.json。

用法(项目根目录):
  python eval/agent_error/schema_probe/probe.py run          # 调 API,12 次
  python eval/agent_error/schema_probe/probe.py summarize    # 只读 results/,写 results/summary.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(AE))
TM = os.environ.get("TOOLMAZE_DIR", os.path.join(ROOT, "external", "agent_error", "toolmaze"))
RUN_DIR = os.path.join(ROOT, "external", "agent_error", "results", "e2", "api", "main")
DEFS = os.path.join(TM, "tools", "definitions")
OUT = os.path.join(HERE, "results")
sys.path.insert(0, AE)
import tool_schema as TS  # noqa: E402

QWEN = "Qwen/Qwen3.5-35B-A3B"
DEEPSEEK = "deepseek-ai/DeepSeek-V3.2"
REPEATS = 3
REQUEST = {"temperature": 0, "max_tokens": 16000, "parallel_tool_calls": False, "timeout": 300,
           "extra_body": {"enable_thinking": False}}             # 与 configs/e2_api.yaml 相同
VARIANTS = {"A": ("原 schema", QWEN), "B": ("additionalProperties: true", QWEN),
            "C": ("按参考补 properties", QWEN), "D": ("原 schema,DeepSeek-V3.2", DEEPSEEK)}


# ---------------------------------------------------------------- 选请求


def pick_episode(run_dir: str, bare: frozenset[tuple[str, str]]) -> dict:
    """基线组推理文件按路径排序,取第一个满足条件的调用:某个顶层无属性对象参数传成 {},
    且此前所有工具结果都没有被注入(请求里只有干净数据)。"""
    for f in sorted(glob.glob(os.path.join(run_dir, "baseline", "*", "fc", "*", "inferences", "*_inference.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        clean = True
        for i, m in enumerate(d["messages"]):
            if m.get("role") == "tool" and (m.get("metadata") or {}).get("perturbation_status") == "perturbed":
                clean = False
            tc = m.get("tool_call")
            if not (clean and tc):
                continue
            empty = [k for k, v in (tc.get("arguments") or {}).items() if (tc["name"], k) in bare and v == {}]
            if empty:
                return {"file": os.path.relpath(f, ROOT), "task_id": d["task_id"], "mode": d["mode"],
                        "message_index": i, "tool": tc["name"], "empty_params": empty, "logged_arguments": tc["arguments"]}
    raise SystemExit("没找到符合条件的调用")


# ---------------------------------------------------------------- 重建请求


def upstream_setup(task_json: dict, api_key: str, base_url: str):
    """用上游代码得到工具定义与系统提示(工具按名字排序:上游从 set 取,原运行里的顺序随进程而变)。"""
    sys.path.insert(0, TM)
    from evaluation.agents.openai_agent import OpenAIAgent
    from evaluation.core.sandbox import ExecutionEngine
    agent = OpenAIAgent(model=QWEN, api_key=api_key, base_url=base_url)
    engine = ExecutionEngine(task_json, agent, tools_dir=os.path.join(TM, "tools"))
    agent.perturbation_mode = engine.mode
    agent.initialize(engine.task_description, engine.tool_definitions)
    tools = sorted(agent.tools, key=lambda t: t["function"]["name"])
    return engine.task_description, agent.messages[0], tools


def replay_messages(system: dict, logged: list[dict], upto: int) -> list[dict]:
    """推理日志里 upto 之前的消息 → OpenAI 格式(助手的 tool_calls 与 tool 结果,与上游 receive_tool_result 相同)。"""
    msgs = [system, {"role": "user", "content": logged[0]["content"]}]
    for m in logged[1:upto]:
        if m.get("role") == "assistant" and m.get("tool_call"):
            tc = m["tool_call"]
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [{
                "id": tc["id"], "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False)}}]})
        elif m.get("role") == "tool":
            msgs.append({"role": "tool", "tool_call_id": m["call_id"], "name": m.get("name"),
                         "content": json.dumps(m.get("content"), ensure_ascii=False)})
        elif m.get("role") == "assistant":
            msgs.append({"role": "assistant", "content": m.get("content") or ""})
    return msgs


def lookup(value: object, path: str) -> object:
    """按 /a/b、/a[] 取示例值;取不到返回 None。"""
    for seg in path.strip("/").split("/"):
        is_item = seg.endswith("[]")
        key = seg[:-2] if is_item else seg
        value = value.get(key) if isinstance(value, dict) else None
        if is_item:
            value = value[0] if isinstance(value, list) and value else None
        if value is None:
            return None
    return value


def reference_properties(p0_task: dict):
    """变体 C:每个工具取 P0 参考里该工具的第一次调用参数作示例,给无属性对象补 properties;找不到示例的不动。"""
    examples = {}
    for vp in p0_task.get("valid_paths", []):
        for s in vp.get("execution_trace", []):
            examples.setdefault(s.get("tool_name"), s.get("arguments") or {})

    def for_tool(name: str):
        def fill(path: str, s: dict) -> dict:
            ex = lookup(examples.get(name, {}), path)
            return {**s, **TS.schema_from_example(ex)} if isinstance(ex, dict) and ex else s
        return fill

    def transform(tools: list[dict]) -> list[dict]:
        out = []
        for t in tools:
            name = t["function"]["name"]
            out += TS.map_tools([t], lambda p, n=name: TS.map_bare_objects(p, for_tool(n)))
        return out
    return transform


def build_requests(ep: dict, api_key: str, base_url: str) -> dict[str, dict]:
    tier = ep["task_id"].split("_")[0].lower()
    with open(os.path.join(TM, "data", "perturbed_tasks", tier, f"{ep['task_id']}_{ep['mode']}.json"), encoding="utf-8") as fh:
        task = json.load(fh)
    with open(os.path.join(TM, "data", "perturbed_tasks", tier, f"{ep['task_id']}_P0.json"), encoding="utf-8") as fh:
        p0 = json.load(fh)
    with open(os.path.join(ROOT, ep["file"]), encoding="utf-8") as fh:
        logged = json.load(fh)["messages"]
    prompt, system, tools = upstream_setup(task, api_key, base_url)
    if prompt != logged[0]["content"]:
        raise SystemExit("重建的用户提示与推理日志不一致")
    messages = replay_messages(system, logged, ep["message_index"])
    schemas = {"A": tools, "B": TS.map_tools(tools, TS.allow_additional_properties),
               "C": reference_properties(p0)(tools), "D": tools}
    return {v: {"model": VARIANTS[v][1], "messages": messages, "tools": schemas[v]} for v in VARIANTS}


# ---------------------------------------------------------------- 运行


def load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def run() -> None:
    from openai import OpenAI
    from toolmaze_run import call_with_retry
    load_env(os.environ.get("ENV_FILE", os.path.join(ROOT, ".env")))     # 在 worktree 里跑时用 ENV_FILE 指向主仓库的 .env
    key, base = os.environ.get("SILICONFLOW_API_KEY"), os.environ.get("SILICONFLOW_BASE_URL")
    if not key or not base:
        raise SystemExit(".env 里缺 SILICONFLOW_API_KEY 或 SILICONFLOW_BASE_URL")
    os.makedirs(OUT, exist_ok=True)
    ep = pick_episode(RUN_DIR, TS.load_bare_params(DEFS))
    reqs = build_requests(ep, key, base)
    meta = {"date": dt.date.today().isoformat(), "episode": ep, "request": {k: v for k, v in REQUEST.items()},
            "variants": {v: {"desc": VARIANTS[v][0], "model": r["model"], "tools_sha": digest(r["tools"]),
                             "messages_sha": digest(r["messages"])} for v, r in reqs.items()}}
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    for v, r in reqs.items():
        with open(os.path.join(OUT, f"request_{v}.json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=1)
    client = OpenAI(api_key=key, base_url=base, max_retries=0)
    with open(os.path.join(OUT, "calls.jsonl"), "w", encoding="utf-8") as out:
        for v, r in reqs.items():
            for i in range(REPEATS):
                rec = {"variant": v, "model": r["model"], "attempt": i + 1}
                try:
                    resp = call_with_retry(lambda: client.chat.completions.create(
                        model=r["model"], messages=r["messages"], tools=r["tools"], **REQUEST), 8, 5, 120)
                    ch = resp.choices[0]
                    calls = ch.message.tool_calls or []
                    rec.update(finish_reason=ch.finish_reason, content=ch.message.content,
                               tool_calls=[{"name": c.function.name, "arguments_raw": c.function.arguments} for c in calls],
                               usage=resp.usage.model_dump() if resp.usage else None)
                except Exception as e:  # noqa: BLE001  记录下来,不中断其余变体
                    rec["error"] = f"{type(e).__name__}: {e}"[:500]
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(v, i + 1, rec.get("error") or [(c["name"], c["arguments_raw"][:120]) for c in rec["tool_calls"]])


# ---------------------------------------------------------------- 汇总


def state(value: object) -> str:
    if value == {}:
        return "空 {}"
    if isinstance(value, dict):
        return "非空对象"
    return "缺失" if value is None else f"其他({type(value).__name__})"


def summarize() -> str:
    with open(os.path.join(OUT, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(OUT, "calls.jsonl"), encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    ep = meta["episode"]
    target, params = ep["tool"], ep["empty_params"]
    lines = [
        "# 无属性对象参数的最小复现(由 eval/agent_error/schema_probe/probe.py summarize 生成,勿手改)", "",
        f"- 日期 {meta['date']};请求取自 `{ep['file']}` 第 {ep['message_index']} 条消息之前"
        f"(任务 {ep['task_id']} {ep['mode']},基线组;此前工具结果都未注入)。",
        f"- ① 原运行里这一步调用 `{target}`,参数 {', '.join(f'`{p}`' for p in params)} 传成 {{}}。",
        f"- 请求参数:{json.dumps(meta['request'], ensure_ascii=False)};每个变体 {REPEATS} 次。", "",
        f"| 变体 | 模型 | 调用的工具 | {' | '.join(f'`{p}`' for p in params)} | 原始 arguments(截断) |",
        "|---" * (4 + len(params)) + "|",
    ]
    for r in recs:
        desc = f"{r['variant']} {meta['variants'][r['variant']]['desc']} #{r['attempt']}"
        if r.get("error"):
            lines.append(f"| {desc} | {r['model']} | 错误 | {' | '.join('—' for _ in params)} | {r['error'][:80]} |")
            continue
        calls = r.get("tool_calls") or []
        if not calls:
            lines.append(f"| {desc} | {r['model']} | 无工具调用({r.get('finish_reason')}) | {' | '.join('—' for _ in params)} | |")
            continue
        c = calls[0]
        try:
            args = json.loads(c["arguments_raw"] or "{}")
        except json.JSONDecodeError:
            args = None
        cells = [state(args.get(p)) if isinstance(args, dict) and c["name"] == target else "—" for p in params]
        raw = (c["arguments_raw"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {desc} | {r['model']} | `{c['name']}` | {' | '.join(cells)} | `{raw[:160]}` |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize", "show"])
    a = ap.parse_args()
    if a.cmd == "run":
        run()
        print(summarize())
    elif a.cmd == "summarize":
        print(summarize())
    else:  # 只选请求、不调用 API
        print(json.dumps(pick_episode(RUN_DIR, TS.load_bare_params(DEFS)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
