"""eval/agent_error/judge/compare.py:多次运行的规则层判定汇总(按结果判分的完成率上下界)。不读 external/。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "agent_error"))
from judge import compare as CMP  # noqa: E402
from judge.rules import Verdict  # noqa: E402

PASS = Verdict("L2", "pass", "pass")
FAIL = Verdict("L2", "fail", "missing_action")
L3 = Verdict("L3", None, "l3", ("untraced",))


def _run():
    recs = {("p1", "baseline"): {"passed": True}, ("p1", "full"): {"passed": True},
            ("p2", "baseline"): {"passed": False}, ("p2", "full"): {"passed": True},
            ("p3", "baseline"): {"passed": False}, ("p3", "full"): {"passed": False}}
    verdicts = {("p1", "baseline"): PASS, ("p1", "full"): L3,
                ("p2", "baseline"): FAIL, ("p2", "full"): PASS,
                ("p3", "baseline"): L3, ("p3", "full"): FAIL}
    return recs, verdicts


def test_arm_counts_split_rule_outcomes_and_toolmaze_passes():
    recs, verdicts = _run()
    c = CMP.arm_counts(recs, verdicts)
    assert c["baseline"] == {"n": 3, "toolmaze_pass": 1, "rule_pass": 1, "rule_fail": 1, "l3": 1}
    assert c["full"] == {"n": 3, "toolmaze_pass": 2, "rule_pass": 1, "rule_fail": 1, "l3": 1}


def test_bounds_treat_l3_as_unknown():
    lo, hi = CMP.bounds({"n": 80, "toolmaze_pass": 10, "rule_pass": 4, "rule_fail": 60, "l3": 16})
    assert (lo, hi) == (4, 20)


def test_outcome_table_has_one_row_per_run_and_arm():
    recs, verdicts = _run()
    lines = CMP.outcome_table({"旧": CMP.arm_counts(recs, verdicts), "新": CMP.arm_counts(recs, verdicts)})
    rows = [ln for ln in lines if ln.startswith("| 旧") or ln.startswith("| 新")]
    assert len(rows) == 4
    assert "| 旧 | 基线 | 1/3 = 33.3%" in rows[0]


def test_reason_table_counts_by_run_arm_and_toolmaze_result():
    recs, verdicts = _run()
    lines = CMP.reason_table({"新": (recs, verdicts)})
    row = next(ln for ln in lines if ln.startswith("| L2 全部满足"))
    # 列:新 基线 ToolMaze 失败 / 通过,新 回执 ToolMaze 失败 / 通过
    assert row.split("|")[2:6] == [" 0 ", " 1 ", " 0 ", " 1 "]
