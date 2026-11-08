"""E2 汇总(summarize_e2.py):回执三分类、反应分类、回执与轨迹对齐、配对统计、清单过滤与输入超长标记。"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "agent_error"))
import contracts as C  # noqa: E402
import run_e2 as E  # noqa: E402
import summarize_e2 as S  # noqa: E402


def call(i, name, args=None):
    return {"role": "assistant", "type": "tool_call", "tool_call": {"id": f"c{i}", "name": name, "arguments": args or {}}}


def res(i, name, content, perturbed=False):
    return {"role": "tool", "name": name, "call_id": f"c{i}", "content": content,
            "metadata": {"perturbation_status": "perturbed" if perturbed else "clean"}}


def test_result_class_three_way():
    assert S.result_class(res(1, "t", {"price": -1}, perturbed=True)) == "injected"
    assert S.result_class(res(1, "t", {"error": "price_data.price_usd is required"})) == "tool_error"
    assert S.result_class(res(1, "t", {"price": 3})) == "normal"
    assert S.result_class(res(1, "t", {"price": 3, "error": ""})) == "normal"          # 空 error 不算


def test_response_retry_switch_other():
    ev = [{"tool": "quote"}, {"tool": "quote"}, {"tool": "quote_alt"}, {"tool": "book"}]
    assert S.response(ev, 0, ["quote", "quote_alt"]) == "retry"
    assert S.response(ev, 1, ["quote", "quote_alt"]) == "switch"
    assert S.response(ev, 2, ["quote_alt"]) == "other"                                 # 下一个工具不在回执列表里
    assert S.response(ev, 3, ["book"]) == "other"                                      # 没有后续调用


def _contract():
    rows = [{"price": float(i)} for i in (1, 2, 3)]
    return C.learn([C.Sample(f"w{i}", "quote", {}, r) for i, r in enumerate(rows)])["quote"]


def test_align_receipts_matches_in_order_and_rejects_mismatch():
    reg = {"quote": _contract()}
    ev = [{"tool": "quote", "args": {}, "result": {"price": 2.0}, "cls": "normal"},
          {"tool": "quote", "args": {}, "result": {"price": -1.0}, "cls": "injected"}]
    rec = {"tool": "quote", "receipt": {}}
    assert S.align_receipts(ev, reg, [rec]) == [(1, rec)]
    with pytest.raises(ValueError):
        S.align_receipts(ev, reg, [])
    with pytest.raises(ValueError):
        S.align_receipts(ev, reg, [{"tool": "other", "receipt": {}}])


def _record(pair, arm, passed, attempt=1, fail=0, mx=1000, tier="c3"):
    return {"pair": pair, "arm": arm, "attempt": attempt, "passed": passed, "api_failures": fail, "tier": tier,
            "max_prompt_tokens": mx, "error_in_answer": False, "harness_error": None}


def test_build_end_to_end_on_synthetic_run():
    pairs = [E.Pair("c3", "C3_task_001", "P3"), E.Pair("c3", "C3_task_002", "P3"), E.Pair("c4", "C4_task_003", "P4")]
    records = [
        _record("C3_task_001_P3", "baseline", False), _record("C3_task_001_P3", "full", True),           # 胜
        _record("C3_task_002_P3", "baseline", True), _record("C3_task_002_P3", "full", True, mx=7000),   # 都过,超长
        _record("C4_task_003_P4", "baseline", False), _record("C4_task_003_P4", "full", False, fail=1),  # API 错误
        _record("C3_task_099_P3", "baseline", False), _record("C3_task_099_P3", "full", True),           # 试跑,不在清单
    ]
    win_traj = {"messages": [call(1, "quote"), res(1, "quote", {"price": -1.0}, perturbed=True),
                             call(2, "quote"), res(2, "quote", {"price": 2.0})]}
    tie_traj = {"messages": [call(1, "quote"), res(1, "quote", {"error": "bad args"}),
                             call(2, "book"), res(2, "book", {"ok": 1})]}
    inferences = {("C3_task_001_P3", "full"): win_traj, ("C3_task_002_P3", "full"): tie_traj}
    receipts = {("C3_task_001_P3", 1): [{"tool": "quote", "receipt": {"admissible_recovery_tools": ["quote", "q2"]}}],
                ("C3_task_002_P3", 1): [{"tool": "quote", "receipt": {"admissible_recovery_tools": ["quote"]}}]}
    folds = {"C3_task_001": 0, "C3_task_002": 1, "C4_task_003": 2}
    regs = [{"quote": _contract()} for _ in range(5)]
    s = S.build(records, pairs, inferences, receipts, (regs, folds), prompt_budget=6144)

    assert s["n_manifest"] == 3 and s["done"] == 2 and s["unresolved"] == ["C4_task_003_P4"]
    assert (s["primary"]["n"], s["primary"]["wins"], s["primary"]["losses"]) == (2, 1, 0)
    assert s["cls_n"] == {"injected": 1, "normal": 2, "tool_error": 1}
    assert s["cls_r"] == {"injected": 1, "tool_error": 1}
    assert s["cls_ge2"]["injected"] == 1 and s["cls_ge2"]["tool_error"] == 0
    assert s["resp_win"]["injected"]["retry"] == 1 and s["resp_all"]["tool_error"]["other"] == 1
    assert s["win_label"] == {"retry": 1}
    assert s["overflow"] == [("C3_task_002_P3", "full", 7000)]
    text = S.render(s, "test")
    assert "C3_task_099" not in text and "胜)1 对" in text
