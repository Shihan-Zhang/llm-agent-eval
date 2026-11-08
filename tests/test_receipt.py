"""E2 回执(receipt.py):信封格式、备用工具、对照变体,以及"不能变成先知"的信息边界。不调用模型。"""

import copy
import inspect
import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "agent_error"))
import contracts as C  # noqa: E402
import receipt as R  # noqa: E402

REAL_DEFS = os.path.join(ROOT, "external", "agent_error", "toolmaze", "tools", "definitions")


def _contract():
    rows = [{"price": 10.0 * i, "currency": "USD", "id": f"S{i}"} for i in (1, 2, 3, 4)]
    return C.learn([C.Sample(f"w{i}", "quote", {"id": r["id"]}, r) for i, r in enumerate(rows)])["quote"]


SUBS = {"quote": ("quote_alpha", "quote_hidden", "quote_yahoo")}
VISIBLE = ["quote", "quote_yahoo", "quote_alpha", "book"]


def bad():
    return {"price": -3.0, "currency": "XXX", "id": "S2"}


# ---------------------------------------------------------------- 构造 Subs(τ)
@pytest.fixture
def defs(tmp_path):
    (tmp_path / "source.yaml").write_text(yaml.safe_dump({"tools": [
        {"name": "w_open", "substitutes": ["w_api", "w_vc"]},
        {"name": "w_api", "substitutes": []},
        {"name": "lonely"}]}), encoding="utf-8")
    (tmp_path / "alternatives.yaml").write_text(yaml.safe_dump({"alternatives": {"many_to_many": {"g1": {"paths": [
        {"path_id": "a", "tools": [{"tool_name": "a1"}, {"tool_name": "a2"}]},
        {"path_id": "b", "tools": [{"tool_name": "b1"}, {"tool_name": "b2"}]},
        {"path_id": "c", "tools": [{"tool_name": "c1"}, {"tool_name": "a2"}]}]}}}}), encoding="utf-8")
    return str(tmp_path)


def test_build_subs_from_substitutes_and_sibling_entry_tools(defs):
    s = R.build_subs(defs)
    assert s["w_open"] == ("w_api", "w_vc")
    assert s["w_api"] == () and s["lonely"] == ()
    assert s["a1"] == ("b1", "c1")                      # 兄弟路径的入口工具
    assert s["b2"] == ("a1", "c1")
    assert s["a2"] == ("a1", "b1", "c1")                # 出现在两条路径里:并集,且不含自身
    assert all(t not in v for t, v in s.items())


def test_build_subs_ignores_files_other_than_definitions(defs, tmp_path):
    before = R.build_subs(defs)
    (tmp_path / "C3_task_001_P3.json").write_text(json.dumps({"alternative_tools": ["leak"]}), encoding="utf-8")
    assert R.build_subs(defs) == before


def test_recovery_tools_puts_faulted_tool_first_and_filters_invisible():
    assert R.recovery_tools("quote", SUBS, VISIBLE) == ["quote", "quote_alpha", "quote_yahoo"]
    assert R.recovery_tools("book", SUBS, VISIBLE) == ["book"]          # 没有备用工具时只能重试自身


# ---------------------------------------------------------------- 信封格式
def test_clean_result_returns_the_same_object():
    r = {"price": 20.0, "currency": "USD", "id": "S2"}
    out = R.augment("quote", {"id": "S2"}, r, _contract(), SUBS, VISIBLE)
    assert out.model_view is r and out.receipt is None and out.violations == ()


def test_tool_without_contract_is_unchanged():
    r = {"anything": -1}
    assert R.augment("unknown", {}, r, None, SUBS, VISIBLE).model_view is r


def test_violation_wraps_raw_result_without_copy_or_mutation():
    r = bad()
    snapshot = copy.deepcopy(r)
    out = R.augment("quote", {"id": "S2"}, r, _contract(), SUBS, VISIBLE)
    assert set(out.model_view) == {"tool_result", "outcome_contract"}
    assert out.model_view["tool_result"] is r and r == snapshot
    assert json.dumps(out.model_view["tool_result"]) == json.dumps(snapshot)


def test_receipt_is_paper_v2_first_violation_five_tuple():
    out = R.augment("quote", {"id": "S2"}, bad(), _contract(), SUBS, VISIBLE)
    rc = out.model_view["outcome_contract"]
    assert set(rc) == {"status", "violation", "admissible_recovery_tools"}
    assert rc["status"] == "inconsistent"
    assert set(rc["violation"]) == {"tool", "property", "path", "expected", "observed"}
    first = out.violations[0]
    assert rc["violation"] == {"tool": "quote", "property": first.property, "path": first.path,
                               "expected": first.expected, "observed": first.observed}
    assert {v.property for v in out.violations} >= {"nonpositive_value", "unseen_category"}   # 全部违反留给日志
    assert rc["admissible_recovery_tools"] == ["quote", "quote_alpha", "quote_yahoo"]
    json.dumps(out.model_view, ensure_ascii=False)                                            # 可序列化


def test_stripped_variant_only_drops_recovery_tools():
    full = R.augment("quote", {"id": "S2"}, bad(), _contract(), SUBS, VISIBLE, "full")
    stripped = R.augment("quote", {"id": "S2"}, bad(), _contract(), SUBS, VISIBLE, "stripped")
    assert "admissible_recovery_tools" not in stripped.receipt
    rest = {k: v for k, v in full.receipt.items() if k != "admissible_recovery_tools"}
    assert stripped.receipt == rest and stripped.violations == full.violations


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        R.augment("quote", {}, bad(), _contract(), SUBS, VISIBLE, "always_warn")


# ---------------------------------------------------------------- 不能变成先知
def test_augment_and_check_accept_no_task_information():
    assert list(inspect.signature(R.augment).parameters) == [
        "tool", "args", "result", "contract", "subs", "visible", "variant"]
    assert list(inspect.signature(C.check).parameters) == ["contract", "args", "result"]


def test_output_is_identical_for_perturbed_and_clean_steps_with_same_result():
    """题目文件里 is_perturbed 标记只存在于步骤上,不会传给检查器;同样的结果必须得到同样的输出。"""
    out_value = bad()
    perturbed_step = {"tool_name": "quote", "arguments": {"id": "S2"}, "output": out_value, "is_perturbed": True}
    clean_step = {"tool_name": "quote", "arguments": {"id": "S2"}, "output": copy.deepcopy(out_value)}
    a = R.augment(perturbed_step["tool_name"], perturbed_step["arguments"], perturbed_step["output"],
                  _contract(), SUBS, VISIBLE)
    b = R.augment(clean_step["tool_name"], clean_step["arguments"], clean_step["output"], _contract(), SUBS, VISIBLE)
    assert a.model_view == b.model_view and a.violations == b.violations


def test_task_like_keys_in_arguments_do_not_change_the_decision():
    clean = {"price": 20.0, "currency": "USD", "id": "S2"}
    leaky = {"id": "S2", "is_perturbed": True, "perturbation_mode": "P3", "expected_result": {"price": 20.0}}
    for result in (clean, bad()):
        a = R.augment("quote", {"id": "S2"}, result, _contract(), SUBS, VISIBLE)
        b = R.augment("quote", leaky, result, _contract(), SUBS, VISIBLE)
        assert a.model_view == b.model_view


def test_receipt_never_names_an_invisible_tool():
    for visible in ([], ["quote"], ["quote_hidden"], VISIBLE, ["book", "quote_alpha"]):
        out = R.augment("quote", {"id": "S2"}, bad(), _contract(), SUBS, visible)
        assert set(out.receipt["admissible_recovery_tools"]) <= set(visible) | {"quote"}


def test_expected_values_come_from_the_learned_contract_only():
    """回执里的期望值来自训练轨迹学到的契约,而不是任何标准答案。"""
    out = R.augment("quote", {"id": "S2"}, {"price": 20.0, "currency": "EUR", "id": "S2"}, _contract(), SUBS, VISIBLE)
    v = out.model_view["outcome_contract"]["violation"]
    assert (v["property"], v["expected"], v["observed"]) == ("unseen_category", ["usd"], "EUR")


# ---------------------------------------------------------------- 真实工具定义(本机有 ToolMaze 时)
@pytest.mark.skipif(not os.path.isdir(REAL_DEFS), reason="需要 external/agent_error/toolmaze")
def test_real_definitions_subs():
    s = R.build_subs(REAL_DEFS)
    assert {"get_weather_weatherapi", "get_weather_visualcrossing"} <= set(s["get_weather_openweather"])
    assert sum(1 for v in s.values() if v) >= 52
