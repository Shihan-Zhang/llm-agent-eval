"""
规则层(L1 / L2)在多次运行上的判定对照,写 eval/results/e2_judge_compare.md(生成文件,勿手改)。不调用模型。

按结果判分的完成率在 L3 裁判之前只有上下界:
  下界 = L2 判通过 / 回合数(L3 全算失败);上界 = (L2 判通过 + L3) / 回合数(L3 全算通过)。

用法(仓库根目录):python eval/agent_error/judge/compare.py
  默认对照 ① 旧管线(e2/api/main)与 ① 新管线(e2/api_v2/main);--run 名字=目录 可重复,覆盖默认。
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
from judge import survey as SV  # noqa: E402
from summarize_e2 import rate  # noqa: E402

RESULTS = os.path.join(ROOT, "external", "agent_error", "results", "e2")
DEFAULT_RUNS = (("① 旧管线", os.path.join(RESULTS, "api", "main")),
                ("① v2", os.path.join(RESULTS, "api_v2", "main")))
OUT = os.path.join(ROOT, "eval", "results", "e2_judge_compare.md")
ARMS = (("baseline", "基线"), ("full", "完整回执"))


def arm_counts(recs: dict, verdicts: dict) -> dict[str, dict[str, int]]:
    out = {arm: {"n": 0, "toolmaze_pass": 0, "rule_pass": 0, "rule_fail": 0, "l3": 0} for arm, _ in ARMS}
    for (pair, arm), v in verdicts.items():
        c = out[arm]
        c["n"] += 1
        c["toolmaze_pass"] += bool(recs[(pair, arm)]["passed"])
        c["l3" if v.level == "L3" else "rule_pass" if v.outcome == "pass" else "rule_fail"] += 1
    return out


def bounds(c: dict[str, int]) -> tuple[int, int]:
    return c["rule_pass"], c["rule_pass"] + c["l3"]


def outcome_table(runs: dict[str, dict]) -> list[str]:
    lines = ["| 运行 | 组 | ToolMaze 路径层通过 | 规则判通过(下界) | 规则判失败 | L3 待裁判 | 上界(L3 全算通过) |",
             "|---|---|---|---|---|---|---|"]
    for name, counts in runs.items():
        for arm, label in ARMS:
            c = counts[arm]
            lo, hi = bounds(c)
            lines.append(f"| {name} | {label} | {rate(c['toolmaze_pass'], c['n'])} | {rate(lo, c['n'])} | "
                         f"{c['rule_fail']} | {c['l3']} | {rate(hi, c['n'])} |")
    return lines


def reason_table(runs: dict[str, tuple[dict, dict]]) -> list[str]:
    cols = [(name, arm, tm) for name in runs for arm, _ in ARMS for tm in (False, True)]
    labels = {a: lab for a, lab in ARMS}
    head = " | ".join(f"{n} {labels[a]} · ToolMaze {'通过' if tm else '失败'}" for n, a, tm in cols)
    tab = collections.Counter()
    for name, (recs, verdicts) in runs.items():
        for (pair, arm), v in verdicts.items():
            tab[(v.reason, name, arm, bool(recs[(pair, arm)]["passed"]))] += 1
    lines = [f"| 判定 | {head} |", "|---" * (len(cols) + 1) + "|"]
    for reason, label in SV.LEVEL_NAMES.items():
        lines.append(f"| {label} | " + " | ".join(str(tab[(reason, *col)]) for col in cols) + " |")
    lines.append("| 合计 | " + " | ".join(str(sum(tab[(r, *col)] for r in SV.LEVEL_NAMES)) for col in cols) + " |")
    return lines


def l3_issue_lines(runs: dict[str, tuple[dict, dict]]) -> list[str]:
    from judge.rules import ISSUE_NAMES
    lines = []
    for name, (_, verdicts) in runs.items():
        c = collections.Counter(i for v in verdicts.values() if v.level == "L3" for i in v.issues)
        lines.append(f"- {name}:" + (";".join(f"{ISSUE_NAMES[i]} {n}" for i, n in c.most_common()) or "无 L3"))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="规则层判定对照(多次运行)")
    ap.add_argument("--run", action="append", default=None, help="名字=运行目录,可重复")
    ap.add_argument("--toolmaze", default=SV.TOOLMAZE)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    specs = [tuple(s.split("=", 1)) for s in a.run] if a.run else list(DEFAULT_RUNS)
    judged = {}
    for name, run_dir in specs:
        recs = SV.load_run(run_dir)
        judged[name] = (recs, SV.judge_all(a.toolmaze, run_dir, recs))
    counts = {name: arm_counts(*rv) for name, rv in judged.items()}
    lines = ["# E2 规则层判定对照(由 eval/agent_error/judge/compare.py 生成,勿手改)", "",
             "运行:" + ";".join(f"{n} = `{os.path.relpath(d, ROOT)}`" for n, d in specs), "",
             "按结果判分在 L3(LLM 裁判)之前只有上下界:下界把 L3 全算失败,上界全算通过。区间为 Wilson 95%。", "",
             "## 按结果判分的完成率", ""]
    lines += outcome_table(counts)
    lines += ["", "## 规则层落点(按 ToolMaze 路径层结果分列)", ""]
    lines += reason_table(judged)
    lines += ["", "## L3 回合的问题(一个回合可有多个,取问题最少的有效路径)", ""]
    lines += l3_issue_lines(judged)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
