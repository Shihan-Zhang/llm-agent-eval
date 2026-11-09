"""
汇总 E2 一次运行(如 API ① 或本地 main),写 eval/results/e2_<api|local>_<run>_summary.md。

只统计冻结清单(configs/e2_manifest.json)里该运行的配对;试跑目录与清单之外的记录不进入结果。
每个(配对, 组)取最近一次尝试;只用两组都完成且没有 API 错误的配对。

- 主要比较:配对完成率,对照组 vs 完整回执(① ③ 的对照是基线,② 的对照是去掉备用工具列表),
  精确 McNemar(只看结果不一致的配对)与配对差的区间;另按层级 × 模式分表。
- 回执三分类(按工具结果):注入的故障(perturbation_status = perturbed)/ 工具返回错误(未注入但结果带
  非空 error 字段,含模型自己传错参数)/ 正常结果上的误报。报每类的结果数、回执数、触发率,
  以及回执里可选工具 ≥ 2 个的占比。
- 收到回执后的反应(按回执事件):下一次工具调用是同一个工具 → 重试原工具;是回执列出的其他工具 → 换用备用工具;
  其他工具或没有后续调用 → 其他。胜出配对(只有回执组完成)按回执三分类再拆;配对级标签取"第一张注入故障回执"后的反应。
- 回执与轨迹对齐:用同一份注册表重新检查回执组轨迹里的每个工具结果(检查是确定性的),
  被标记的结果按顺序与 receipts.jsonl 的记录一一对应;工具名对不上即报错。
- 本地模型:列出单次请求最大输入超过 num_ctx − max_tokens 的回合(Ollama 超出上下文时静默截断)。
- 敏感性分析:本运行里有同一配置试跑(<results_dir>/pilot/order.json)用过的任务时,去掉这些配对重算主要比较。
- 复现性(① 以外的运行):完整回执组与 ① 的完整回执组在共同配对上逐对比较,报一致率、Cohen κ 和 2×2 表。

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
ARM_NAMES = {"baseline": "基线", "full": "完整回执", "stripped": "去掉备用工具列表"}


def compared_arms(run_cfg: dict) -> tuple[str, str]:
    """(对照组, 处理组)。处理组固定是完整回执;① ③ 的对照是基线,② 的对照是去掉备用工具列表的回执。"""
    arms = list(run_cfg["arms"])
    if len(arms) != 2 or TREATED not in arms:
        raise ValueError(f"运行必须恰好两组且含 {TREATED!r},收到 {arms}")
    control = next(a for a in arms if a != TREATED)
    if control not in ARM_NAMES:
        raise ValueError(f"未知的对照组 {control!r}")
    return control, TREATED


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
          prompt_budget: int | None, control: str = BASELINE) -> dict:
    """inferences[(pair, arm)] = 推理 JSON;receipts[(pair, attempt)] = 完整回执组该次尝试的回执记录列表。
    回执与反应只分析完整回执组(② 的对照组回执不带备用工具列表,"换用备用工具"无定义)。"""
    keys = {p.key: p for p in manifest_pairs}
    latest = {(r["pair"], r["arm"]): r for r in E.latest_records([r for r in records if r["pair"] in keys])}
    done, unresolved = [], []
    for k in keys:
        rb, rt = latest.get((k, control)), latest.get((k, TREATED))
        ok = rb and rt and rb["attempt"] == rt["attempt"] and rb["api_failures"] == 0 and rt["api_failures"] == 0
        (done if ok else unresolved).append(k)
    regs, folds = folds_regs
    cls_n, cls_r, cls_ge2 = Counter(), Counter(), Counter()
    resp_all = defaultdict(Counter)        # 类别 → 反应 → 回执事件数(全部配对)
    resp_win = defaultdict(Counter)        # 同上,只看胜出配对
    win_label = Counter()
    strata = defaultdict(list)
    rows = []
    outcomes = {}                          # 配对 → (对照组通过, 完整回执组通过)
    for k in done:
        p = keys[k]
        x, y = bool(latest[(k, control)]["passed"]), bool(latest[(k, TREATED)]["passed"])
        rows.append((x, y))
        outcomes[k] = (x, y)
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
            "win_label": win_label, "overflow": overflow, "prompt_budget": prompt_budget, "control": control,
            "outcomes": outcomes}


def pct_ci(lo: float, hi: float) -> str:
    return f"[{max(-1.0, lo):.1%}, {min(1.0, hi):.1%}]"


def diff_lines(p: dict, cn: str = ARM_NAMES[BASELINE], tn: str = ARM_NAMES[TREATED]) -> list[str]:
    """配对差 δ = (胜 − 负)/n 的点估计与 95% 区间;cn / tn = 对照组 / 处理组的名称。"""
    if not p["n"]:
        return []
    out = [f"配对差 δ = (胜 − 负)/n = ({p['wins']} − {p['losses']})/{p['n']} = {p['diff']:.1%}"
           f"(= {tn}组完成率 − {cn}组完成率)。", ""]
    if p["diff_wilson"] is not None:
        out += [f"- Wilson {pct_ci(*p['diff_wilson'])}:负 = 0 时 δ = 胜/n,取 Wilson(胜, n),"
                f"即 (p̂ + z²/2n ± z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n),p̂ = {p['wins']}/{p['n']},z = 1.96;"
                "它把负的配对数当作确定的 0,没有计入负的抽样不确定性,所以下界偏乐观。"]
    out += [f"- Newcombe 方法 10 {pct_ci(*p['diff_newcombe'])}:两组完成率各取 Wilson 区间,按配对相关 φ 合成"
            "(公式见 summarize_e2.newcombe_paired);任何胜 / 负组合都适用,作为配对差的主区间。", ""]
    return out


def render(s: dict, title: str) -> str:
    p = s["primary"]
    cn, tn = ARM_NAMES[s["control"]], ARM_NAMES[TREATED]
    lines = [f"# {title}(由 eval/agent_error/summarize_e2.py 生成,勿手改)", "",
             "## 运行完整性", "",
             f"- 冻结清单配对 {s['n_manifest']} 对;两组都完成且无 API 错误 {s['done']} 对;未完成 {len(s['unresolved'])} 对"
             + (f"({', '.join(s['unresolved'][:10])})" if s["unresolved"] else ""),
             f"- 因 API 错误或中断整对重跑 {s['pair_reruns']} 次;最终回答含 \"Error during execution\" 的回合 "
             f"{s['error_in_answer']} 个(多为模型参数无法解析等非 API 原因);运行框架报错 {s['harness_errors']} 个", "",
             f"## 主要比较:配对完成率({cn} vs {tn})", "",
             "| 组 | 完成率 [95% Wilson] |", "|---|---|",
             f"| {cn} | {rate(p['base'], p['n'])} |", f"| {tn} | {rate(p['treat'], p['n'])} |", "",
             f"不一致配对:只有{tn}组完成(胜){p['wins']} 对,只有{cn}组完成(负){p['losses']} 对;"
             f"精确 McNemar 双侧 p = {p['p']:.3g}。", ""] + diff_lines(p, cn, tn) + [
             f"| 层级 | 模式 | 配对 | {cn} | {tn} | 胜 : 负 | p |", "|---|---|---|---|---|---|---|"]
    for (tier, mode), q in s["strata"].items():
        lines.append(f"| {tier.upper()} | {mode} | {q['n']} | {q['base']} | {q['treat']} | {q['wins']} : {q['losses']} | {q['p']:.3g} |")
    total_r = sum(s["cls_r"].values())
    lines += ["", f"## 回执三分类({tn}组全部工具结果)", "",
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
    lines += extra_lines(s, cn, tn)
    return "\n".join(lines) + "\n"


def sensitivity(outcomes: dict, pairs: list[E.Pair], excluded_tasks: set[str]) -> dict | None:
    """去掉 excluded_tasks(如试跑用过的任务)后重算配对比较;本运行里没有这些任务时返回 None。"""
    drop = {p.key for p in pairs if p.task_id in excluded_tasks and p.key in outcomes}
    if not drop:
        return None
    rows = [v for k, v in outcomes.items() if k not in drop]
    return {"dropped": sorted(drop), **paired(rows)}


def cohen_kappa(rows: list[tuple[bool, bool]]) -> float | None:
    """两次运行逐对通过与否的 Cohen κ = (p_o − p_e)/(1 − p_e);p_e = 1(两边全同一类)时无定义,返回 None。"""
    n = len(rows)
    if not n:
        return None
    po = sum(a == b for a, b in rows) / n
    p1, p2 = sum(a for a, _ in rows) / n, sum(b for _, b in rows) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return None if pe == 1 else (po - pe) / (1 - pe)


def replication(outcomes: dict, reference: dict[str, bool]) -> dict | None:
    """完整回执组与参考运行(① 的完整回执组)在共同配对上逐对比较:一致率、κ、2×2 表。"""
    rows = [(reference[k], y) for k, (_, y) in sorted(outcomes.items()) if k in reference]
    if not rows:
        return None
    return {"n": len(rows), "agree": sum(a == b for a, b in rows), "kappa": cohen_kappa(rows),
            "both": sum(a and b for a, b in rows), "ref_only": sum(a and not b for a, b in rows),
            "new_only": sum(b and not a for a, b in rows), "neither": sum(not a and not b for a, b in rows),
            "p": mcnemar_exact(sum(b and not a for a, b in rows), sum(a and not b for a, b in rows))}


def extra_lines(s: dict, cn: str, tn: str) -> list[str]:
    out = []
    sen = s.get("sensitivity")
    if sen:
        out += ["", "## 敏感性分析:去掉与试跑重叠的任务", "",
                f"去掉 {len(sen['dropped'])} 对({', '.join(sen['dropped'])};这些任务用作过试跑,结果未计入),剩 {sen['n']} 对:", "",
                "| 组 | 完成率 [95% Wilson] |", "|---|---|",
                f"| {cn} | {rate(sen['base'], sen['n'])} |", f"| {tn} | {rate(sen['treat'], sen['n'])} |", "",
                f"胜 : 负 = {sen['wins']} : {sen['losses']},精确 McNemar 双侧 p = {sen['p']:.3g};"
                f"配对差 δ = {sen['diff']:.1%},Newcombe {pct_ci(*sen['diff_newcombe'])}。"]
    rep = s.get("replication")
    if rep:
        kappa = "无定义(两次运行全部同为通过或同为失败)" if rep["kappa"] is None else f"{rep['kappa']:.2f}"
        out += ["", f"## 复现性:{tn}组与 ① 的{tn}组逐对比较(共同的 {rep['n']} 对)", "",
                f"| | 本次通过 | 本次失败 |", "|---|---|---|",
                f"| ① 通过 | {rep['both']} | {rep['ref_only']} |", f"| ① 失败 | {rep['new_only']} | {rep['neither']} |", "",
                f"一致率 {rate(rep['agree'], rep['n'])};Cohen κ = {kappa};"
                f"两次运行通过率之差的精确 McNemar 双侧 p = {rep['p']:.3g}。",
                "同一配置、温度 0 重跑,不一致来自 API 端的非确定性;κ 衡量这种抖动下逐题结论有多稳定。"]
    return out


def pilot_tasks(results_dir: str) -> set[str]:
    """同一配置试跑用过的任务(<results_dir>/pilot/order.json);没有试跑记录时为空集。"""
    path = os.path.join(results_dir, "pilot", "order.json")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {e["task_id"] for e in json.load(fh)["order"]}


def reference_full(results_dir: str, ref_run: str = "main") -> dict[str, bool]:
    """参考运行(①)里完整回执组每对最近一次、无 API 错误的通过与否。"""
    recs = E.read_records(os.path.join(results_dir, ref_run, "episodes.jsonl"))
    return {r["pair"]: bool(r["passed"]) for r in E.latest_records(recs)
            if r["arm"] == TREATED and r["api_failures"] == 0}


# ---------------------------------------------------------------- 读文件
def load_run(cfg: dict, cfg_path: str, run: str) -> tuple[str, list[dict], dict, dict, list[E.Pair]]:
    e2 = cfg["e2"]
    out = os.path.abspath(os.path.join(os.path.dirname(cfg_path), e2["results_dir"], run))
    records = E.read_records(os.path.join(out, "episodes.jsonl"))
    pairs = E.pairs_for(e2["runs"][run]["episodes"], M.load())
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
    control, _ = compared_arms(e2["runs"][a.run])
    s = build(records, pairs, inferences, receipts, C.load(os.path.join(C.OUT, e2.get("registry", "gold"))), budget,
              control)
    results_dir = os.path.dirname(out)
    s["sensitivity"] = sensitivity(s["outcomes"], pairs, pilot_tasks(results_dir))
    s["replication"] = replication(s["outcomes"], reference_full(results_dir) if a.run != "main" else {})
    src = "local" if e2.get("num_ctx") else "api"
    text = render(s, f"E2 {src} / {a.run}:{cfg['agent']['model']}")
    path = os.path.join(ROOT, "eval", "results", f"e2_{src}_{a.run}_summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
