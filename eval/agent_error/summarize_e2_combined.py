"""
把同一配置下的几次 E2 运行(如 ① v2 的 main 80 对 + main_supp 120 对)合并成一份配对比较,
写 eval/results/e2_<tag>_<name>_summary.md(生成文件,勿手改)。不调用模型。

每次运行先按 summarize_e2.build 取「两组都完成且无 API 错误」的配对(与单次汇总口径相同),再合并:
- 合计:两组完成率、胜 : 负、精确 McNemar、配对差 δ 与 Newcombe 区间(summarize_e2.paired)。
- 分层:按运行(清单)、按层级 × 模式(合计,并列出各运行的胜 : 负)。
- 一致性检验:各运行的效果是否一致——只看不一致配对,比较「胜 /(胜 + 负)」,两次运行时用 Fisher 精确检验(双侧)。
  配对设计下,McNemar 只依赖不一致配对,这个比例相同等价于各部分的配对优势比相同。
- 多重检验:第一次运行单独检验过一次,合并后又检验一次;两次检验按 Bonferroni 校正(p × 2,上限 1)。

用法:python eval/agent_error/summarize_e2_combined.py --config eval/agent_error/configs/e2_api_v2.yaml \
        --runs main,main_supp --name main200
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import contracts as C  # noqa: E402
import run_e2 as E  # noqa: E402
import summarize_e2 as S  # noqa: E402

N_TESTS = 2                     # 第一次运行单独检验 + 合并后检验


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """2×2 表 [[a, b], [c, d]] 的 Fisher 精确检验双侧 p:固定边缘,累加概率不超过观测表的所有表。"""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    if r1 == 0 or r2 == 0 or c1 == 0 or c1 == n:
        return 1.0
    total = math.comb(n, c1)

    def prob(x: int) -> float:
        return math.comb(r1, x) * math.comb(r2, c1 - x) / total

    observed = prob(a)
    p = sum(prob(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if prob(x) <= observed * (1 + 1e-9))
    return min(1.0, p)


def bonferroni(p: float, m: int) -> float:
    if m < 1:
        raise ValueError(f"检验次数必须 ≥ 1,收到 {m}")
    return min(1.0, p * m)


def combine(parts: dict[str, tuple[list[E.Pair], dict[str, tuple[bool, bool]]]]) -> dict:
    """parts[运行名] = (该运行清单里的配对, outcomes[配对] = (对照组通过, 完整回执组通过));运行之间不得有重复配对。"""
    seen, rows, by_part = {}, [], {}
    strata, strata_parts = defaultdict(list), defaultdict(list)
    for name, (pairs, outcomes) in parts.items():
        meta = {p.key: p for p in pairs}
        for k in outcomes:
            if k in seen:
                raise ValueError(f"配对 {k} 同时出现在 {seen[k]} 和 {name}")
            seen[k] = name
        part_rows = []
        for k, o in sorted(outcomes.items()):
            p = meta[k]
            part_rows.append(o)
            strata[(p.tier, p.mode)].append(o)
            strata_parts[(p.tier, p.mode, name)].append(o)
        rows += part_rows
        by_part[name] = S.paired(part_rows)
    out = {"total": S.paired(rows), "parts": by_part,
           "strata": {s: S.paired(v) for s, v in sorted(strata.items())},
           "strata_parts": {s: S.paired(v) for s, v in sorted(strata_parts.items())}, "homogeneity": None}
    if len(by_part) == 2:
        (w1, l1), (w2, l2) = ((q["wins"], q["losses"]) for q in by_part.values())
        out["homogeneity"] = fisher_exact(w1, l1, w2, l2)
    return out


def _delta(q: dict) -> str:
    return f"{q['diff']:.1%} {S.pct_ci(*q['diff_newcombe'])}" if q["n"] else "—"


def render(c: dict, title: str, first: str, decision: str, cn: str = S.ARM_NAMES[S.BASELINE],
           tn: str = S.ARM_NAMES[S.TREATED]) -> str:
    t, names = c["total"], list(c["parts"])
    p1 = c["parts"][first]["p"]
    merged = " + ".join(f"{n}({q['n']} 对)" for n, q in c["parts"].items())
    lines = [f"# {title}(由 eval/agent_error/summarize_e2_combined.py 生成,勿手改)", "",
             f"合并的运行:{merged} = {t['n']} 对。", "",
             f"**事后扩样**:{decision}", "",
             f"## 合计:配对完成率({cn} vs {tn})", "",
             "| 组 | 完成率 [95% Wilson] |", "|---|---|",
             f"| {cn} | {S.rate(t['base'], t['n'])} |", f"| {tn} | {S.rate(t['treat'], t['n'])} |", "",
             f"不一致配对:胜 {t['wins']} 对、负 {t['losses']} 对;精确 McNemar 双侧 p = {t['p']:.3g}。", ""]
    lines += S.diff_lines(t, cn, tn)
    lines += ["## 多重检验(Bonferroni)", "",
              f"同一假设检验了 {N_TESTS} 次:先在 {first} 单独检验,看到结果后扩样,再在合并的 {t['n']} 对上检验。"
              f"按 Bonferroni 把每个 p 乘以 {N_TESTS}(上限 1),对应总的第一类错误率不超过 5%。", "",
              "| 检验 | 配对 | 胜 : 负 | 原始 p | Bonferroni 校正 p |", "|---|---|---|---|---|",
              f"| 第 1 次:{first} | {c['parts'][first]['n']} | {c['parts'][first]['wins']} : {c['parts'][first]['losses']} | "
              f"{p1:.3g} | {bonferroni(p1, N_TESTS):.3g} |",
              f"| 第 2 次:合计 | {t['n']} | {t['wins']} : {t['losses']} | {t['p']:.3g} | {bonferroni(t['p'], N_TESTS):.3g} |",
              "", "## 按清单分层", "",
              f"| 清单 | 配对 | {cn} | {tn} | 胜 : 负 | p | δ [Newcombe] |", "|---|---|---|---|---|---|---|"]
    for n in names:
        q = c["parts"][n]
        lines.append(f"| {n} | {q['n']} | {q['base']} | {q['treat']} | {q['wins']} : {q['losses']} | {q['p']:.3g} | {_delta(q)} |")
    lines.append(f"| 合计 | {t['n']} | {t['base']} | {t['treat']} | {t['wins']} : {t['losses']} | {t['p']:.3g} | {_delta(t)} |")
    lines += ["", "## 按层级 × 模式分层(合计,另列各清单的胜 : 负)", "",
              f"| 层级 | 模式 | 配对 | {cn} | {tn} | 胜 : 负 | p | " + " | ".join(f"{n} 胜 : 负" for n in names) + " |",
              "|---|---|---|---|---|---|---|" + "---|" * len(names)]
    for (tier, mode), q in c["strata"].items():
        per = []
        for n in names:
            s = c["strata_parts"].get((tier, mode, n))
            per.append(f"{s['wins']} : {s['losses']}" if s else "—")
        lines.append(f"| {tier.upper()} | {mode} | {q['n']} | {q['base']} | {q['treat']} | {q['wins']} : {q['losses']} | "
                     f"{q['p']:.3g} | " + " | ".join(per) + " |")
    if c["homogeneity"] is not None:
        (n1, q1), (n2, q2) = c["parts"].items()
        lines += ["", "## 各清单效果是否一致", "",
                  f"只看不一致配对,比较胜的比例:{n1} {q1['wins']}/{q1['wins'] + q1['losses']},"
                  f"{n2} {q2['wins']}/{q2['wins'] + q2['losses']};Fisher 精确检验(双侧)p = {c['homogeneity']:.3g}。",
                  "这个比例相同等价于两部分的配对优势比相同(McNemar 只依赖不一致配对)。"
                  "不一致配对少时检验功效很低:p 大只说明没有检出差异,不说明两部分效果相同。"]
    return "\n".join(lines) + "\n"


def run_outcomes(cfg: dict, cfg_path: str, run: str) -> tuple[list[E.Pair], dict, str]:
    """与 summarize_e2.main 相同的口径算出一次运行的 outcomes;返回(清单配对, outcomes, 对照组)。"""
    e2 = cfg["e2"]
    _, records, inferences, receipts, pairs = S.load_run(cfg, cfg_path, run)
    if not records:
        sys.exit(f"运行 {run} 没有记录")
    control, _ = S.compared_arms(e2["runs"][run])
    s = S.build(records, pairs, inferences, receipts, C.load(os.path.join(C.OUT, e2.get("registry", "gold"))), None, control)
    if s["unresolved"]:
        print(f"[combined] 注意:{run} 有 {len(s['unresolved'])} 对未完成,不计入", file=sys.stderr)
    return pairs, s["outcomes"], control


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", required=True, help="逗号分隔;第一个是先单独检验过的那次运行")
    ap.add_argument("--name", required=True, help="汇总文件名里的运行名,如 main200")
    a = ap.parse_args()
    cfg_path = os.path.abspath(a.config)
    cfg = E.load_cfg(cfg_path)
    runs = a.runs.split(",")
    parts, controls = {}, set()
    for run in runs:
        pairs, outcomes, control = run_outcomes(cfg, cfg_path, run)
        parts[run] = (pairs, outcomes)
        controls.add(control)
    if len(controls) != 1:
        sys.exit(f"各运行的对照组不同:{controls}")
    control = controls.pop()
    c = combine(parts)
    first = runs[0]
    decision = (f"扩到 {c['total']['n']} 对的决定是在看到 {first} {c['parts'][first]['n']} 对的结果"
                f"(胜 : 负 {c['parts'][first]['wins']} : {c['parts'][first]['losses']},p = {c['parts'][first]['p']:.3g})"
                "之后做的,不是事先计划的样本量;补充配对来自运行前冻结的补充清单(configs/e2_manifest_ext.json),"
                "清单本身没有因结果改动。所以合计的 p 要做多重检验校正,见下文。")
    src = cfg["e2"].get("tag") or "api"
    text = render(c, f"E2 {src} / {a.name}:{cfg['agent']['model']}", first, decision, S.ARM_NAMES[control])
    path = os.path.join(ROOT, "eval", "results", f"e2_{src}_{a.name}_summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
