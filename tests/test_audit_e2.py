"""E2 离线审计(audit_e2.py):第 k 折只能用 C_k、故障/干净结果的抽取与去重、汇总口径。不读真实数据。"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "agent_error"))
import audit_e2 as A  # noqa: E402
import contracts as C  # noqa: E402


class OnlyThisFold(dict):
    """其他折的注册表:任何访问都报错,用来证明审计没有碰它。"""

    def get(self, *a, **k):
        raise AssertionError("审计用了不属于该题所在折的注册表")

    __getitem__ = get
    __contains__ = get


def _contract():
    rows = [{"price": float(i)} for i in (1, 2, 3)]
    return C.learn([C.Sample(f"w{i}", "quote", {}, r) for i, r in enumerate(rows)])["quote"]


def _outcome(task_id, result, faulty=True, tool="quote", visible=("quote",)):
    return A.Outcome(task_id, "c3", "P3" if faulty else "P0", tool, {}, result, faulty, "logic", visible)


def test_audit_uses_only_the_registry_of_the_tasks_fold():
    folds = {"C3_task_001": 2, "C3_task_002": 4}
    regs = [OnlyThisFold() for _ in range(5)]
    regs[2] = {"quote": _contract()}
    rows = A.audit([_outcome("C3_task_001", {"price": -1.0})], regs, folds, {})
    assert rows[0].fold == 2 and rows[0].covered and rows[0].flagged
    with pytest.raises(AssertionError):
        A.audit([_outcome("C3_task_002", {"price": -1.0})], regs, folds, {})


def test_contract_in_other_folds_does_not_count_as_coverage():
    folds = {"C3_task_001": 1}
    regs = [{"quote": _contract()} for _ in range(5)]
    regs[1] = {}
    row, = A.audit([_outcome("C3_task_001", {"price": -1.0})], regs, folds, {})
    assert not row.covered and not row.flagged


def _step(tool, out, perturbed=False):
    s = {"tool_name": tool, "arguments": {"id": 1}, "output": out, "status": "success"}
    return {**s, "is_perturbed": True} if perturbed else s


def test_outcomes_keep_only_perturbed_steps_dedupe_across_paths_and_union_visible_tools():
    bad = _step("quote", {"price": -1}, perturbed=True)
    task = {"task_id": "C4_task_001", "perturbation_mode": "P4", "corruption_type": "logic_violation",
            "valid_paths": [{"execution_trace": [_step("a", {}), bad, _step("quote", {"price": 2})]},
                            {"execution_trace": [_step("b", {}), bad]}]}
    outs = A.outcomes_from_task(task, "c4")
    assert [(o.tool, o.result, o.faulty) for o in outs] == [("quote", {"price": -1}, True)]
    assert outs[0].visible == ("a", "b", "quote") and outs[0].corruption_type == "logic_violation"


def test_p0_outcomes_are_clean_steps():
    task = {"task_id": "C3_task_001", "perturbation_mode": "P0",
            "valid_paths": [{"execution_trace": [_step("a", {"x": 1}), _step("b", {"y": 2})]},
                            {"execution_trace": [_step("a", {"x": 1})]}]}
    outs = A.outcomes_from_task(task, "c3")
    assert [(o.tool, o.faulty) for o in outs] == [("a", False), ("b", False)]


def test_stats_definitions():
    def row(faulty, flagged, covered=True, n_rec=1):
        v = (C.Violation("nonpositive_value", "p", "> 0", -1, ""),) if flagged else ()
        return A.Row(_outcome("t", {}, faulty), 0, covered, v, tuple(f"r{i}" for i in range(n_rec)))
    rows = [row(True, True, n_rec=2), row(True, True, n_rec=1), row(True, False), row(True, False, covered=False),
            row(False, True), row(False, False), row(False, False), row(False, False)]
    s = A.stats(rows)
    assert (s["n_fault"], s["recall"], s["covered"], s["recall_covered"]) == (4, 2, 3, 2)
    assert (s["n_clean"], s["fp"]) == (4, 1)
    assert (s["n_flag_fault"], s["ge2"]) == (2, 1)
