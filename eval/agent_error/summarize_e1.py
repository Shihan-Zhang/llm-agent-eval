"""
从 ToolMaze 的结果目录重算 E1 报告里的数字,写成 eval/results/e1_summary.md。

上游 run_eval.py 只算 TSR / PRR / RC(写在 metrics/metrics_report_*.json);这里补上它不算的:
- 每个模式成功率的 95% Wilson 区间;
- 事先指定的主要比较:隐式(P3、P4)vs 显式(P1、P2),按(题, 暂时/持续)配对,
  即 (P1, P3) 与 (P2, P4) 两组,每题 2 对;只看结果不一致的对,精确 McNemar;
- 命中率:回合内至少有一次工具结果被注入故障;
- "声称重试过但没有":最终回答提到 retry,但至少有一个被注入故障的工具只调用了 1 次。
  这一条是描述性的启发式规则(只认英文 retry 的各种词形),清单列在报告里,便于逐条人工复核。

判分沿用上游写好的 evaluations/*_eval.json,这里不重新判分。

用法(仓库根目录):
    python eval/agent_error/summarize_e1.py [--results external/agent_error/results/e1/qwen3i-8k/fc/c1]
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "eval", "fc"))
from summarize_fc import mcnemar_exact, wilson  # noqa: E402

RESULTS = os.path.join(ROOT, "external", "agent_error", "results", "e1", "qwen3i-8k", "fc", "c1")
OUT = os.path.join(ROOT, "eval", "results", "e1_summary.md")
MODES = ["P0", "P1", "P2", "P3", "P4"]
PAIRS = [("P1", "P3"), ("P2", "P4")]  # (显式, 隐式):暂时一对、持续一对
RETRY = re.compile(r"\bretr(y|ied|ies|ying)\b", re.I)


def _key(path, suffix):
    task_id, mode = os.path.basename(path)[: -len(suffix)].rsplit("_", 1)
    return task_id, mode


def load(results_dir):
    """返回 {(task_id, mode): pass}、{(task_id, mode): inference} 和最新的上游指标报告(可能为 None)。"""
    passed, inference = {}, {}
    for f in glob.glob(os.path.join(results_dir, "evaluations", "*_eval.json")):
        with open(f, encoding="utf-8") as fh:
            passed[_key(f, "_eval.json")] = bool(json.load(fh)["pass"])
    for f in glob.glob(os.path.join(results_dir, "inferences", "*_inference.json")):
        with open(f, encoding="utf-8") as fh:
            inference[_key(f, "_inference.json")] = json.load(fh)
    reports = sorted(glob.glob(os.path.join(results_dir, "metrics", "metrics_report_*.json")))
    metrics = None
    if reports:
        with open(reports[-1], encoding="utf-8") as fh:
            metrics = json.load(fh)
    return passed, inference, metrics


def paired_counts(passed):
    """主要比较:返回 (对数, 只有显式通过的对数, 只有隐式通过的对数)。缺任一边的题跳过。"""
    n = explicit_only = implicit_only = 0
    for task_id in sorted({t for t, _ in passed}):
        for explicit, implicit in PAIRS:
            if (task_id, explicit) not in passed or (task_id, implicit) not in passed:
                continue
            e, i = passed[(task_id, explicit)], passed[(task_id, implicit)]
            n += 1
            explicit_only += e and not i
            implicit_only += i and not e
    return n, explicit_only, implicit_only


def victim_tools(messages):
    return {m["name"] for m in messages
            if m.get("role") == "tool" and (m.get("metadata") or {}).get("perturbation_status") == "perturbed"}


def final_answer(messages):
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("type") != "tool_call":
            return m.get("content") or ""
    return ""


def unbacked_retry_claim(messages):
    """最终回答说重试过,但有被注入故障的工具只调用了 1 次:返回这些工具名(排序),否则空列表。"""
    victims = victim_tools(messages)
    if not victims or not RETRY.search(final_answer(messages)):
        return []
    calls = Counter(m["tool_call"]["name"] for m in messages if m.get("type") == "tool_call")
    return sorted(v for v in victims if calls[v] < 2)


def build_report(passed, inference, metrics, results_dir):
    lines = ["# E1 汇总(由 eval/agent_error/summarize_e1.py 生成,勿手改)", "",
             f"结果目录:`{os.path.relpath(results_dir, ROOT).replace(os.sep, '/')}`;"
             f"判分文件 {len(passed)} 个,推理文件 {len(inference)} 个。", "",
             "| 模式 | 通过 / 总数 | TSR [95% Wilson] | 命中率 | PRR(上游) | RC(上游) |",
             "|---|---|---|---|---|---|"]
    for mode in MODES:
        keys = [k for k in passed if k[1] == mode]
        k = sum(passed[x] for x in keys)
        lo, hi = wilson(k, len(keys))
        hits = sum(bool(victim_tools(inference[x]["messages"])) for x in keys if x in inference)
        hit = "—" if mode == "P0" else f"{hits}/{len(keys)}"
        prr = rc = "—"
        if metrics and mode != "P0":
            prr = f"{metrics['prr'][mode]:.1%}" if metrics["prr"].get(mode) is not None else "—"
            rc = f"{metrics['rc'][mode]:.2f}" if metrics["rc"].get(mode) is not None else "—"
        lines.append(f"| {mode} | {k}/{len(keys)} | {k / len(keys):.0%} [{lo:.0%}, {hi:.0%}] | {hit} | {prr} | {rc} |")

    n, b, c = paired_counts(passed)
    lines += ["", "## 主要比较(事先指定):隐式 vs 显式", "",
              f"配对 {n} 对((P1, P3) 与 (P2, P4),每题各 1 对)。只有显式通过 {b} 对,只有隐式通过 {c} 对;"
              f"精确 McNemar 双侧 p = {mcnemar_exact(b, c):.2e}(= 2 × P(Binomial({b + c}, 1/2) ≤ {min(b, c)}))。"]

    claims = []
    for (task_id, mode), d in sorted(inference.items()):
        tools = unbacked_retry_claim(d["messages"])
        if mode != "P0" and tools:
            claims.append((mode, task_id, tools))
    claims.sort()
    per_mode = Counter(m for m, _, _ in claims)
    lines += ["", "## 声称重试过但没有(描述性,启发式)", "",
              "规则:最终回答匹配 `retr(y|ied|ies|ying)`,且至少一个被注入故障的工具只调用了 1 次。"
              f"共 {len(claims)} 个回合:" + "、".join(f"{m} {per_mode[m]}" for m in MODES if per_mode[m]) + "。", "",
              "| 模式 | 回合 | 只调用 1 次的故障工具 |", "|---|---|---|"]
    lines += [f"| {m} | {t} | {', '.join(tools)} |" for m, t, tools in claims]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    passed, inference, metrics = load(a.results)
    if not passed:
        sys.exit(f"没有找到判分文件:{a.results}")
    report = build_report(passed, inference, metrics, os.path.abspath(a.results))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
