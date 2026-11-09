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


def test_paired_difference_intervals_when_no_losses():
    rows = [(False, True)] * 11 + [(False, False)] * 69
    p = S.paired(rows)
    assert p["diff"] == pytest.approx(11 / 80)
    lo, hi = p["diff_wilson"]
    assert (round(lo, 4), round(hi, 4)) == (0.0785, 0.2297)          # 报告里的 [7.9%, 23.0%]
    # 某边缘为 0 → φ = 0:L = δ − √((p₁−l₁)² + (u₂−p₂)²),U = u₁(基线 0/80 时 p₂ = l₂ = 0)
    l1, u1 = S.wilson(11, 80)
    _, u2 = S.wilson(0, 80)
    nlo, nhi = p["diff_newcombe"]
    assert nlo == pytest.approx(11 / 80 - ((11 / 80 - l1) ** 2 + u2 ** 2) ** 0.5)
    assert nhi == pytest.approx(u1)
    assert nlo < lo                                                   # Newcombe 计入了负的不确定性,更宽


def test_paired_difference_wilson_only_without_losses_and_newcombe_symmetric():
    p = S.paired([(False, True)] * 5 + [(True, False)] * 5 + [(True, True)] * 10 + [(False, False)] * 30)
    assert p["diff"] == 0 and p["diff_wilson"] is None
    lo, hi = p["diff_newcombe"]
    assert lo < 0 < hi and lo == pytest.approx(-hi)


def test_render_shows_paired_difference():
    rows = [(False, True)] * 11 + [(False, False)] * 69
    text = "\n".join(S.diff_lines(S.paired(rows)))
    assert "13.8%" in text and "Wilson [7.9%, 23.0%]" in text and "Newcombe 方法 10 [6.3%, 23.0%]" in text


def test_stripped_run_compares_full_against_stripped():
    pairs = [E.Pair("c3", "C3_task_001", "P3"), E.Pair("c3", "C3_task_002", "P3")]
    records = [
        _record("C3_task_001_P3", "stripped", False), _record("C3_task_001_P3", "full", True),     # 胜:只有完整回执完成
        _record("C3_task_002_P3", "stripped", True), _record("C3_task_002_P3", "full", False),     # 负
    ]
    traj = {"messages": [call(1, "book"), res(1, "book", {"ok": 1})]}
    inferences = {("C3_task_001_P3", "full"): traj, ("C3_task_002_P3", "full"): traj}
    folds = {"C3_task_001": 0, "C3_task_002": 1}
    regs = [{} for _ in range(5)]
    s = S.build(records, pairs, inferences, {}, (regs, folds), prompt_budget=None, control="stripped")
    assert s["done"] == 2 and (s["primary"]["wins"], s["primary"]["losses"]) == (1, 1)
    text = S.render(s, "test")
    assert "去掉备用工具列表 vs 完整回执" in text and "基线" not in text


def test_compared_arms_from_real_configs():
    api = E.load_cfg(os.path.join(os.path.dirname(S.HERE), "agent_error", "configs", "e2_api.yaml"))["e2"]["runs"]
    assert S.compared_arms(api["main"]) == ("baseline", "full")
    assert S.compared_arms(api["p0"]) == ("baseline", "full")
    assert S.compared_arms(api["stripped"]) == ("stripped", "full")
    with pytest.raises(ValueError):
        S.compared_arms({"arms": {"baseline": "off", "stripped": "stripped"}})
