"""
汇总 E2 一次运行(如 API ① 或本地 main),写 eval/results/e2_<api|local>_<run>_summary.md。

只统计冻结清单(configs/e2_manifest.json)里该运行的配对;试跑目录与清单之外的记录不进入结果。
每个(配对, 组)取最近一次尝试;只用两组都完成且没有 API 错误的配对。

- 主要比较:配对完成率,基线 vs 完整回执,精确 McNemar(只看结果不一致的配对);另按层级 × 模式分表。
- 回执三分类(按工具结果):注入的故障(perturbation_status = perturbed)/ 工具返回错误(未注入但结果带
  非空 error 字段,含模型自己传错参数)/ 正常结果上的误报。报每类的结果数、回执数、触发率,
  以及回执里可选工具 ≥ 2 个的占比。
- 收到回执后的反应(按回执事件):下一次工具调用是同一个工具 → 重试原工具;是回执列出的其他工具 → 换用备用工具;
  其他工具或没有后续调用 → 其他。胜出配对(只有回执组完成)按回执三分类再拆;配对级标签取"第一张注入故障回执"后的反应。
- 回执与轨迹对齐:用同一份注册表重新检查回执组轨迹里的每个工具结果(检查是确定性的),
  被标记的结果按顺序与 receipts.jsonl 的记录一一对应;工具名对不上即报错。
- 本地模型:列出单次请求最大输入超过 num_ctx − max_tokens 的回合(Ollama 超出上下文时静默截断)。

用法:python eval/agent_error/summarize_e2.py --config eval/agent_error/configs/e2_api.yaml --run main
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eval", "fc"))
import contracts as C  # noqa: E402
import e2_manifest as M  # noqa: E402
import run_e2 as E  # noqa: E402
from summarize_fc import mcnemar_exact, wilson  # noqa: E402

CLASSES = ("injected", "tool_error", "normal")
CLASS_NAMES = {"injected": "注入的故障", "tool_error": "工具返回错误", "normal": "正常结果上的误报"}
RESPONSES = ("retry", "switch", "other")
RESPONSE_NAMES = {"retry": "重试原工具", "switch": "换用备用工具", "other": "其他"}
BASELINE, TREATED = "baseline", "full"


# ---------------------------------------------------------------- 轨迹与回执
def result_class(msg: dict) -> str:
    if (msg.get("metadata") or {}).get("perturbation_status") == "perturbed":
        return "injected"
    content = msg.get("content")
    return "tool_error" if isinstance(content, dict) and C.has_error_field(content) else "normal"


def tool_events(inference: dict) -> list[dict]:
    """按顺序列出工具结果:工具名、调用参数、原始结果、三分类。"""
    calls = {m["tool_call"]["id"]: m["tool_call"] for m in inference["messages"] if m.get("type") == "tool_call"}
    out = []
    for m in inference["messages"]:
        if m.get("role") != "tool":
            continue
        call = calls.get(m.get("call_id")) or {}
        out.append({"tool": m["name"], "args": call.get("arguments") or {}, "result": m.get("content"),
                    "cls": result_class(m)})
    return out


def align_receipts(events: list[dict], registry: C.Registry, receipts: list[dict]) -> list[tuple[int, dict]]:
    """返回 [(事件下标, 回执记录)]。重新检查得到的被标记事件必须与回执日志逐条对应。"""
    flagged = [i for i, e in enumerate(events) if C.check(registry.get(e["tool"]), e["args"], e["result"])]
    if len(flagged) != len(receipts) or any(events[i]["tool"] != r["tool"] for i, r in zip(flagged, receipts)):
        raise ValueError(f"回执日志与轨迹不一致:重新检查标记 {len(flagged)} 个,日志 {len(receipts)} 条")
    return list(zip(flagged, receipts))


def response(events: list[dict], i: int, recovery: list[str]) -> str:
    """回执之后的下一次工具调用:同一工具 → retry;回执列出的其他工具 → switch;其余(含没有后续调用)→ other。"""
    if i + 1 >= len(events):
        return "other"
    nxt, tool = events[i + 1]["tool"], events[i]["tool"]
    if nxt == tool:
        return "retry"
    if nxt in recovery:
        return "switch"
    return "other"


# ---------------------------------------------------------------- 汇总
def rate(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [{max(0.0, lo):.1%}, {min(1.0, hi):.1%}]"   # 截掉浮点误差造成的 -0.0%


def newcombe_paired(both: int, wins: int, losses: int, n: int) -> tuple[float, float]:
    """配对差 δ = (胜 − 负)/n 的 95% 区间:Newcombe (1998) 方法 10,两组各自的 Wilson 区间按 φ 合成。
    L = δ − √((p₁−l₁)² − 2φ(p₁−l₁)(u₂−p₂) + (u₂−p₂)²),U = δ + √((u₁−p₁)² − 2φ(u₁−p₁)(p₂−l₂) + (p₂−l₂)²);
    p₁ = 回执组完成率,p₂ = 基线完成率,φ = 2×2 配对表的相关系数(某边缘为 0 时取 0)。"""
    neither = n - both - wins - losses
    p1, p2 = (both + wins) / n, (both + losses) / n
    (l1, u1), (l2, u2) = wilson(both + wins, n), wilson(both + losses, n)
    denom = math.sqrt((both + wins) * (losses + neither) * (both + losses) * (wins + neither))
    phi = (both * neither - wins * losses) / denom if denom else 0.0
    delta = p1 - p2
    lo = delta - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = delta + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, lo), min(1.0, hi)


def paired(rows: list[tuple[bool, bool]]) -> dict:
    """rows = [(基线通过, 回执通过)]。"""
    b = sum(1 for x, y in rows if not x and y)       # 只有回执组完成(胜)
    c = sum(1 for x, y in rows if x and not y)       # 只有基线完成(负)
    both = sum(1 for x, y in rows if x and y)
    n = len(rows)
    out = {"n": n, "base": sum(x for x, _ in rows), "treat": sum(y for _, y in rows),
           "wins": b, "losses": c, "both": both, "p": mcnemar_exact(b, c)}
    if n:
        out["diff"] = (b - c) / n
        out["diff_newcombe"] = newcombe_paired(both, b, c, n)
        out["diff_wilson"] = wilson(b, n) if c == 0 else None   # 负 = 0 时 δ = 胜/n,可用 Wilson(胜, n)
    return out


def build(records: list[dict], manifest_pairs: list[E.Pair], inferences: dict, receipts: dict, folds_regs,
          prompt_budget: int | None) -> dict:
    """inferences[(pair, arm)] = 推理 JSON;receipts[(pair, attempt)] = 回执组该次尝试的回执记录列表。"""
    keys = {p.key: p for p in manifest_pairs}
    latest = {(r["pair"], r["arm"]): r for r in E.latest_records([r for r in records if r["pair"] in keys])}
    done, unresolved = [], []
    for k in keys:
        rb, rt = latest.get((k, BASELINE)), latest.get((k, TREATED))
        ok = rb and rt and rb["attempt"] == rt["attempt"] and rb["api_failures"] == 0 and rt["api_failures"] == 0
        (done if ok else unresolved).append(k)
    regs, folds = folds_regs
    cls_n, cls_r, cls_ge2 = Counter(), Counter(), Counter()
    resp_all = defaultdict(Counter)        # 类别 → 反应 → 回执事件数(全部配对)
    resp_win = defaultdict(Counter)        # 同上,只看胜出配对
    win_label = Counter()
    strata = defaultdict(list)
    rows = []
    for k in done:
        p = keys[k]
        x, y = bool(latest[(k, BASELINE)]["passed"]), bool(latest[(k, TREATED)]["passed"])
        rows.append((x, y))
        strata[(p.tier, p.mode)].append((x, y))
        win = (not x) and y
        events = tool_events(inferences[(k, TREATED)])
        for e in events:
            cls_n[e["cls"]] += 1
        aligned = align_receipts(events, C.registry_for(regs, folds, p.task_id),
                                 receipts.get((k, latest[(k, TREATED)]["attempt"]), []))
        first_injected = None
        for i, rec in aligned:
            cls = events[i]["cls"]
            recovery = rec["receipt"].get("admissible_recovery_tools") or []
            r = response(events, i, recovery)
            cls_r[cls] += 1
            cls_ge2[cls] += len(recovery) >= 2
            resp_all[cls][r] += 1
            if win:
                resp_win[cls][r] += 1
                if cls == "injected" and first_injected is None:
                    first_injected = r
        if win:
            win_label[first_injected or "no_injected_receipt"] += 1
    overflow = []
    if prompt_budget:
        overflow = sorted((r["pair"], r["arm"], r["max_prompt_tokens"]) for r in latest.values()
                          if r["pair"] in keys and r["max_prompt_tokens"] > prompt_budget)
    eps = [r for r in latest.values() if r["pair"] in keys]
    return {"n_manifest": len(keys), "done": len(done), "unresolved": sorted(unresolved),
            "pair_reruns": E.pair_reruns([r for r in records if r["pair"] in keys]),
            "error_in_answer": sum(r["error_in_answer"] for r in eps),
            "harness_errors": sum(bool(r.get("harness_error")) for r in eps),
            "primary": paired(rows), "strata": {s: paired(v) for s, v in sorted(strata.items())},
            "cls_n": cls_n, "cls_r": cls_r, "cls_ge2": cls_ge2, "resp_all": resp_all, "resp_win": resp_win,
            "win_label": win_label, "overflow": overflow, "prompt_budget": prompt_budget}


def pct_ci(lo: float, hi: float) -> str:
    return f"[{max(-1.0, lo):.1%}, {min(1.0, hi):.1%}]"


def diff_lines(p: dict) -> list[str]:
    """配对差 δ = (胜 − 负)/n 的点估计与 95% 区间。"""
    if not p["n"]:
        return []
    out = [f"配对差 δ = (胜 − 负)/n = ({p['wins']} − {p['losses']})/{p['n']} = {p['diff']:.1%}(= 回执组完成率 − 基线完成率)。", ""]
    if p["diff_wilson"] is not None:
        out += [f"- Wilson {pct_ci(*p['diff_wilson'])}:负 = 0 时 δ = 胜/n,取 Wilson(胜, n),"
                f"即 (p̂ + z²/2n ± z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n),p̂ = {p['wins']}/{p['n']},z = 1.96;"
                "它把负的配对数当作确定的 0,没有计入负的抽样不确定性,所以下界偏乐观。"]
    out += [f"- Newcombe 方法 10 {pct_ci(*p['diff_newcombe'])}:两组完成率各取 Wilson 区间,按配对相关 φ 合成"
            "(公式见 summarize_e2.newcombe_paired);任何胜 / 负组合都适用,作为配对差的主区间。", ""]
    return out


def render(s: dict, title: str) -> str:
    p = s["primary"]
    lines = [f"# {title}(由 eval/agent_error/summarize_e2.py 生成,勿手改)", "",
             "## 运行完整性", "",
             f"- 冻结清单配对 {s['n_manifest']} 对;两组都完成且无 API 错误 {s['done']} 对;未完成 {len(s['unresolved'])} 对"
             + (f"({', '.join(s['unresolved'][:10])})" if s["unresolved"] else ""),
             f"- 因 API 错误或中断整对重跑 {s['pair_reruns']} 次;最终回答含 \"Error during execution\" 的回合 "
             f"{s['error_in_answer']} 个(多为模型参数无法解析等非 API 原因);运行框架报错 {s['harness_errors']} 个", "",
             "## 主要比较:配对完成率(基线 vs 完整回执)", "",
             "| 组 | 完成率 [95% Wilson] |", "|---|---|",
             f"| 基线 | {rate(p['base'], p['n'])} |", f"| 完整回执 | {rate(p['treat'], p['n'])} |", "",
             f"不一致配对:只有回执组完成(胜){p['wins']} 对,只有基线完成(负){p['losses']} 对;"
             f"精确 McNemar 双侧 p = {p['p']:.3g}。", ""] + diff_lines(p) + [
             "| 层级 | 模式 | 配对 | 基线 | 回执 | 胜 : 负 | p |", "|---|---|---|---|---|---|---|"]
    for (tier, mode), q in s["strata"].items():
        lines.append(f"| {tier.upper()} | {mode} | {q['n']} | {q['base']} | {q['treat']} | {q['wins']} : {q['losses']} | {q['p']:.3g} |")
    total_r = sum(s["cls_r"].values())
    lines += ["", "## 回执三分类(回执组全部工具结果)", "",
              "| 类别 | 工具结果 | 回执 | 触发率 | 回执中可选工具 ≥ 2 |", "|---|---|---|---|---|"]
    for c in CLASSES:
        lines.append(f"| {CLASS_NAMES[c]} | {s['cls_n'][c]} | {s['cls_r'][c]} | {rate(s['cls_r'][c], s['cls_n'][c])} | "
                     f"{rate(s['cls_ge2'][c], s['cls_r'][c])} |")
    lines.append(f"| 合计 | {sum(s['cls_n'].values())} | {total_r} | {rate(total_r, sum(s['cls_n'].values()))} | "
                 f"{rate(sum(s['cls_ge2'].values()), total_r)}(论文 91.5%) |")
    for name, table in (("全部配对", s["resp_all"]), ("胜出配对", s["resp_win"])):
        lines += ["", f"## 收到回执后的反应({name},按回执事件)", "",
                  "| 回执类别 | " + " | ".join(RESPONSE_NAMES[r] for r in RESPONSES) + " | 合计 |",
                  "|---|" + "---|" * (len(RESPONSES) + 1)]
        for c in CLASSES:
            n = sum(table[c].values())
            lines.append(f"| {CLASS_NAMES[c]} | " + " | ".join(rate(table[c][r], n) for r in RESPONSES) + f" | {n} |")
    wl = s["win_label"]
    n_win = sum(wl.values())
    lines += ["", "**胜出配对的标签**(第一张注入故障回执之后的反应)", "",
              "| 标签 | 配对数 |", "|---|---|"]
    lines += [f"| {RESPONSE_NAMES[r]} | {rate(wl[r], n_win)} |" for r in RESPONSES]
    lines += [f"| 没有收到注入故障的回执 | {rate(wl['no_injected_receipt'], n_win)} |"]
    if s["prompt_budget"]:
        lines += ["", f"## 输入长度(预算 {s['prompt_budget']} token = num_ctx − max_tokens)", "",
                  f"单次请求最大输入超过预算的回合:{len(s['overflow'])} 个" + ("。" if not s["overflow"] else ":")]
        lines += [f"- {pair} {arm}:{mx}" for pair, arm, mx in s["overflow"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 读文件
def load_run(cfg: dict, cfg_path: str, run: str) -> tuple[str, list[dict], dict, dict, list[E.Pair]]:
    e2 = cfg["e2"]
    out = os.path.abspath(os.path.join(os.path.dirname(cfg_path), e2["results_dir"], run))
    records = E.read_records(os.path.join(out, "episodes.jsonl"))
    with open(M.OUT, encoding="utf-8") as fh:
        pairs = E.pairs_for(e2["runs"][run]["episodes"], json.load(fh))
    model = cfg["agent"]["model"].replace("/", "_").replace(":", "_")
    inferences = {}
    for r in E.latest_records(records):
        path = os.path.join(out, r["arm"], model, "fc", r["tier"], "inferences", f"{r['pair']}_inference.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                inferences[(r["pair"], r["arm"])] = json.load(fh)
    receipts = defaultdict(list)
    for rec in E.read_records(os.path.join(out, "receipts.jsonl")):
        if rec.get("arm") == TREATED:
            receipts[(rec["pair"], rec["attempt"])].append(rec)
    return out, records, inferences, receipts, pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    cfg_path = os.path.abspath(a.config)
    cfg = E.load_cfg(cfg_path)
    out, records, inferences, receipts, pairs = load_run(cfg, cfg_path, a.run)
    if not records:
        sys.exit(f"没有运行记录:{out}")
    e2 = cfg["e2"]
    budget = (e2["num_ctx"] - cfg["agent"]["max_tokens"]) if e2.get("num_ctx") else None
    s = build(records, pairs, inferences, receipts, C.load(os.path.join(C.OUT, e2.get("registry", "gold"))), budget)
    src = "local" if e2.get("num_ctx") else "api"
    text = render(s, f"E2 {src} / {a.run}:{cfg['agent']['model']}")
    path = os.path.join(ROOT, "eval", "results", f"e2_{src}_{a.run}_summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
