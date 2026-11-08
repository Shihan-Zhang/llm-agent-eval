"""
E2 阶段 3:离线检测审计(不调用模型)。对照论文 arXiv 2608.19303 §3 的 task-disjoint 审计:
C3/C4 上召回率 79.6%、干净误报率 1.39%、契约覆盖率 94.2%;§5 中回执保留 ≥ 2 个可选恢复动作的占比 91.5%。

口径(与论文一致,按"每个被观察到的工具结果"计,不按回合):
- 故障结果:C3/C4 的 P3/P4 题目文件中带 is_perturbed 的步骤;同一题目文件内相同(工具, 参数, 输出)只计一次
  (多条备选路径共享同一故障步骤)。
- 干净结果:C3/C4 的 P0 题目文件中的步骤(同样去重)。
- 第 k 折的题只用注册表 C_k 检查(contracts.registry_for;C_k 没见过这些题)。
- 覆盖:C_k 里有该工具的契约。召回:检查器至少报出一条违反。
- 可选工具 ≥ 2:回执的 admissible_recovery_tools(出错工具 + 本题可见的备用工具)长度 ≥ 2。
  可见工具照搬上游 ExecutionEngine._get_c2_merged_trace:所有路径执行轨迹里工具名的并集(C1 为 execution_trace)。

用法(仓库根目录,需先运行 contracts.py build --source gold / e1):
    python eval/agent_error/audit_e2.py            # → eval/results/e2_audit.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eval", "fc"))
import contracts as C  # noqa: E402
import receipt as R  # noqa: E402
from summarize_fc import wilson  # noqa: E402

DEFS = os.path.join(C.TM, "tools", "definitions")
OUT = os.path.join(ROOT, "eval", "results", "e2_audit.md")
TIERS = ("c3", "c4")
PAPER = {"recall": 0.796, "fp": 0.0139, "coverage": 0.942, "ge2": 0.915}


@dataclass(frozen=True)
class Outcome:
    task_id: str
    tier: str
    mode: str
    tool: str
    args: dict
    result: dict
    faulty: bool
    corruption_type: str
    visible: tuple[str, ...]


@dataclass(frozen=True)
class Row:
    outcome: Outcome
    fold: int
    covered: bool
    violations: tuple[C.Violation, ...]
    recovery: tuple[str, ...]

    @property
    def flagged(self) -> bool:
        return bool(self.violations)


def _traces(task: dict) -> list[list[dict]]:
    if task.get("valid_paths"):
        return [p.get("execution_trace") or [] for p in task["valid_paths"]]
    return [task.get("execution_trace") or []]


def task_visible_tools(task: dict) -> tuple[str, ...]:
    return tuple(sorted({st["tool_name"] for tr in _traces(task) for st in tr if st.get("tool_name")}))


def outcomes_from_task(task: dict, tier: str) -> list[Outcome]:
    """P0:全部步骤(干净);P1–P4:只取 is_perturbed 的步骤(故障)。同一题目文件内去重。"""
    mode = task.get("perturbation_mode", "P0")
    faulty = mode != "P0"
    visible = task_visible_tools(task)
    seen, out = set(), []
    for tr in _traces(task):
        for st in tr:
            if not isinstance(st.get("output"), dict) or bool(st.get("is_perturbed")) != faulty:
                continue
            key = (st["tool_name"], json.dumps(st.get("arguments") or {}, sort_keys=True),
                   json.dumps(st["output"], sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            out.append(Outcome(task["task_id"], tier, mode, st["tool_name"], st.get("arguments") or {}, st["output"],
                               faulty, task.get("corruption_type") or "", visible))
    return out


def load_outcomes(tm_dir: str = C.TM, tiers: Iterable[str] = TIERS, modes: Iterable[str] = ("P0", "P3", "P4")) -> list[Outcome]:
    out = []
    for tier in tiers:
        for mode in modes:
            for f in sorted(glob.glob(os.path.join(tm_dir, "data", "perturbed_tasks", tier, f"*_{mode}.json"))):
                with open(f, encoding="utf-8") as fh:
                    out += outcomes_from_task(json.load(fh), tier)
    return out


def audit(outcomes: Iterable[Outcome], registries: list[C.Registry], folds: dict[str, int], subs: R.Subs) -> list[Row]:
    """每个结果只用它所在折的注册表 C_k 检查。"""
    rows = []
    for o in outcomes:
        reg = C.registry_for(registries, folds, o.task_id)
        contract = reg.get(o.tool)
        violations = tuple(C.check(contract, o.args, o.result))
        rows.append(Row(o, folds[o.task_id], contract is not None, violations,
                        tuple(R.recovery_tools(o.tool, subs, o.visible))))
    return rows


# ---------------------------------------------------------------- 汇总
def rate(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def stats(rows: list[Row]) -> dict:
    faults = [r for r in rows if r.outcome.faulty]
    clean = [r for r in rows if not r.outcome.faulty]
    flagged_faults = [r for r in faults if r.flagged]
    return {"n_fault": len(faults), "recall": sum(r.flagged for r in faults),
            "covered": sum(r.covered for r in faults), "recall_covered": sum(r.flagged for r in faults if r.covered),
            "n_clean": len(clean), "fp": sum(r.flagged for r in clean),
            "n_flag_fault": len(flagged_faults), "ge2": sum(len(r.recovery) >= 2 for r in flagged_faults)}


def section(name: str, rows: list[Row], k: int) -> list[str]:
    s = stats(rows)
    lines = [f"## 注册表来源:{name}", "",
             "| 指标 | 本次 | 论文 |", "|---|---|---|",
             f"| 召回率(故障结果被标记) | {rate(s['recall'], s['n_fault'])} | 79.6% |",
             f"| 干净误报率(P0 结果被标记) | {rate(s['fp'], s['n_clean'])} | 1.39% |",
             f"| 契约覆盖率(故障结果的工具在 C_k 中有契约) | {rate(s['covered'], s['n_fault'])} | 94.2% |",
             f"| 有契约时的召回率 | {rate(s['recall_covered'], s['covered'])} | — |",
             f"| 被标记的故障中,回执可选工具 ≥ 2 个 | {rate(s['ge2'], s['n_flag_fault'])} | 91.5%(运行中 484 次回执) |",
             "", "**按折**(审计第 k 折只用 C_k)", "",
             "| 折 | 故障结果 | 召回率 | 干净结果 | 误报率 | 覆盖率 |", "|---|---|---|---|---|---|"]
    for fold in range(k):
        f = stats([r for r in rows if r.fold == fold])
        lines.append(f"| {fold} | {f['n_fault']} | {rate(f['recall'], f['n_fault'])} | {f['n_clean']} | "
                     f"{rate(f['fp'], f['n_clean'])} | {rate(f['covered'], f['n_fault'])} |")
    lines += ["", "**按层级与模式**", "",
              "| 层级 | 模式 | 故障结果 | 召回率 | 覆盖率 | 可选工具 ≥ 2 |", "|---|---|---|---|---|---|"]
    for tier in TIERS:
        for mode in ("P3", "P4"):
            f = stats([r for r in rows if r.outcome.tier == tier and r.outcome.mode == mode])
            lines.append(f"| {tier.upper()} | {mode} | {f['n_fault']} | {rate(f['recall'], f['n_fault'])} | "
                         f"{rate(f['covered'], f['n_fault'])} | {rate(f['ge2'], f['n_flag_fault'])} |")
        c = stats([r for r in rows if r.outcome.tier == tier])
        lines.append(f"| {tier.upper()} | 合计 | {c['n_fault']} | {rate(c['recall'], c['n_fault'])} | "
                     f"{rate(c['covered'], c['n_fault'])} | 误报 {rate(c['fp'], c['n_clean'])} |")

    faults = [r for r in rows if r.outcome.faulty]
    fires, first = Counter(), Counter()
    for r in faults:
        for p in {v.property for v in r.violations}:
            fires[(r.outcome.tier, p)] += 1
        if r.violations:
            first[(r.outcome.tier, r.violations[0].property)] += 1
    n_tier = Counter(r.outcome.tier for r in faults)
    lines += ["", "**按违反类别**(某类在故障结果上触发的比例;一条故障可触发多类。"
              "\"回执首条\"是回执里实际展示给模型的那一类)", "",
              "| 违反类别 | C3 触发 | C4 触发 | C3 回执首条 | C4 回执首条 |", "|---|---|---|---|---|"]
    for p in sorted({p for _, p in fires}, key=lambda p: -sum(fires[(t, p)] for t in TIERS)):
        lines.append(f"| {p} | " + " | ".join(rate(fires[(t, p)], n_tier[t]) for t in TIERS) + " | "
                     + " | ".join(str(first[(t, p)]) for t in TIERS) + " |")

    by_ct = defaultdict(list)
    for r in faults:
        by_ct[r.outcome.corruption_type or "(未标注)"].append(r)
    lines += ["", "**按故障类型**(题目文件的 corruption_type;对照论文附录 M 表 19)", "",
              "| 故障类型 | 故障结果 | 召回率 | 覆盖率 |", "|---|---|---|---|"]
    for ct, rs in sorted(by_ct.items(), key=lambda x: -len(x[1])):
        f = stats(rs)
        lines.append(f"| {ct} | {f['n_fault']} | {rate(f['recall'], f['n_fault'])} | {rate(f['covered'], f['n_fault'])} |")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["gold", "e1"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    outcomes = load_outcomes()
    if not outcomes:
        sys.exit(f"没有找到 C3/C4 题目文件:{C.TM}")
    subs = R.build_subs(DEFS)
    lines = ["# E2 阶段 3:离线检测审计(由 eval/agent_error/audit_e2.py 生成,勿手改)", "",
             f"规格哈希 {C.SPEC_SHA256[:12]};范围 C3/C4 × P0(干净)/ P3、P4(故障);按工具结果计,不按回合。"
             "区间为 95% Wilson。", ""]
    for src in a.sources:
        regs, folds = C.load(os.path.join(C.OUT, src))
        with open(os.path.join(C.OUT, src, "build_info.json"), encoding="utf-8") as fh:
            info = json.load(fh)
        lines += section(f"{src}({info['workflows']} 个工作流、{info['samples']} 个样本、"
                         f"{info['invariants_total']:,} 条不变量,仓库 {info['repo_commit']})",
                         audit(outcomes, regs, folds, subs), C.SPEC["folds"])
    text = "\n".join(lines)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
