"""阶段 C 规则层(eval/agent_error/judge):规范化、轨迹解析、L1 / L2 判定。测试数据全部自造,不读 external/。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "agent_error"))
from judge import normalize as N  # noqa: E402
from judge import rules as RU  # noqa: E402
from judge import trajectory as TR  # noqa: E402
from judge.reference import Catalog, task_ref_from_p0  # noqa: E402

# ---------------------------------------------------------------- 规范化


def test_numbers_compare_with_tolerance():
    assert N.same(N.canon(1.0), N.canon(1.004))            # 绝对容差 0.005
    assert not N.same(N.canon(1.0), N.canon(1.006))
    assert N.same(N.canon(28785), N.canon(28787))          # 相对 6.9e-5 ≤ 1e-4
    assert not N.same(N.canon(28785), N.canon(28790))      # 相对 1.7e-4
    assert N.same(N.canon("170.25"), N.canon(170.25))      # 纯数字字符串按数字比


def test_dates_compare_as_iso():
    assert N.same(N.canon("2024-11-15 14:00"), N.canon("2024-11-15T14:00:00"))
    assert N.same(N.canon("Nov 15, 2024"), N.canon("2024/11/15"))
    assert N.same(N.canon("14:00"), N.canon("2024-11-15 14:00"))    # 只有时间时比时间部分
    assert not N.same(N.canon("2024-11-15"), N.canon("2024-11-16"))


@pytest.mark.parametrize("key,value,expected", [
    ("flag", True, "bool"), ("attendees", 2, "small_int"), ("balance", 28785, "number"),
    ("order_id", "ORD-SELL-FAEF1", "id"), ("user_id", "user_01", "id"), ("workspace_path", "/home/user/iot_configs", "path"),
    ("airline", "Air China", "label"), ("route", "Beijing Capital Airport -> Shanghai", "long"), ("date", "2024-11-15", "date"),
])
def test_kind(key, value, expected):
    assert N.kind(key, value) == expected


def test_containment_for_paths_and_dates():
    assert N.contained("iot_configs", "/home/user/iot_configs")
    assert N.contained("2026-02-22 15:04:54", "Current UTC time: 2026-02-22 15:04:54")
    assert not N.contained("abc", "xxabcxx")                          # 太短,不算


# ---------------------------------------------------------------- 自造的工具目录与参考

CATALOG = Catalog(
    category={"get_sheet": "Source", "archive_sheet": "Action", "publish_alt": "Action", "create_note": "Action"},
    required={"get_sheet": frozenset({"user_id"}), "archive_sheet": frozenset({"balance_sheet", "title"}),
              "publish_alt": frozenset({"payload"}), "create_note": frozenset({"title"})},
    substitutes={"archive_sheet": frozenset({"publish_alt"})},
)
SHEET = {"sheet_id": "SHEET-0001", "amount": 28785}
REF = task_ref_from_p0({"valid_paths": [{"execution_trace": [
    {"tool_name": "get_sheet", "arguments": {"user_id": "user_01"}, "output": SHEET},
    {"tool_name": "archive_sheet", "arguments": {"balance_sheet": SHEET, "title": "Q3 Balance", "priority": "high"},
     "output": {"status": "archived", "note_id": "NOTE18E3D"}},
]}]})
PROMPT = "Get the balance sheet for user_01 and archive it as Q3 Balance."


def episode(calls, final="Done."):
    """calls = [(工具, 参数, 输出, 是否注入)];final=None 表示没有最终回答。"""
    msgs = [{"role": "user", "content": PROMPT}]
    for i, (tool, args, out, injected) in enumerate(calls):
        msgs.append({"role": "assistant", "type": "tool_call", "content": "", "tool_call": {"id": f"c{i}", "name": tool, "arguments": args}})
        msgs.append({"role": "tool", "name": tool, "call_id": f"c{i}", "content": out,
                     "metadata": {"perturbation_status": "perturbed" if injected else "clean"}})
    if final is not None:
        msgs.append({"role": "assistant", "type": "text", "content": final})
    return TR.parse(msgs)


GET = ("get_sheet", {"user_id": "user_01"}, SHEET, False)


def archive(sheet=None, title="Q3 Balance", **extra):
    args = {"balance_sheet": SHEET if sheet is None else sheet, "title": title, **extra}
    return ("archive_sheet", args, {"status": "archived", "note_id": "NOTE18E3D"}, False)


def verdict(calls, final="Done."):
    return RU.judge(episode(calls, final), REF, CATALOG)


def test_parse_marks_injection_errors_and_missing_final_answer():
    t = episode([GET, ("get_sheet", {"user_id": "u"}, {"error": "bad"}, False)], final=None)
    assert [c.ok for c in t.calls] == [True, False] and t.final_answer is None
    assert episode([("get_sheet", {}, SHEET, True)]).calls[0].injected


def test_correct_episode_passes_and_optional_params_are_exempt():
    assert verdict([GET, archive(priority="low")]) == RU.Verdict("L2", "pass", "pass")


def test_no_final_answer_still_goes_through_l2():
    assert verdict([GET, archive()], final=None).outcome == "pass"


def test_empty_required_container_fails():               # C3_task_002 型:归档了空表
    assert verdict([GET, archive(sheet={})]).reason == "empty_argument"


def test_empty_container_on_substitute_fails():          # C3_task_039 型:换路后发布了空对象
    v = verdict([GET, ("publish_alt", {"payload": {}}, {"status": "published"}, False)])
    assert v.reason == "empty_argument"


def test_corrupted_value_in_argument_fails():
    bad = {"sheet_id": "SHEET-ERR-9999", "amount": -1}
    v = verdict([("get_sheet", {"user_id": "user_01"}, bad, True), archive(sheet={"sheet_id": "SHEET-ERR-9999", "amount": 28785})])
    assert v.reason == "corrupted_argument"


def test_corrupted_value_in_final_answer_is_l1():
    bad = {"sheet_id": "SHEET-ERR-9999", "amount": -1}
    v = verdict([("get_sheet", {"user_id": "user_01"}, bad, True), GET, archive()], final="Archived SHEET-ERR-9999.")
    assert (v.level, v.reason) == ("L1", "answer_corrupted")


def test_injected_output_is_not_a_source():
    same_values_but_injected = ("get_sheet", {"user_id": "user_01"}, {"sheet_id": "SHEET-0001", "amount": -1}, True)
    v = verdict([same_values_but_injected, archive()])
    assert v.level == "L3" and "untraced" in v.issues


def test_source_must_appear_before_use():
    v = verdict([archive(), GET])
    assert v.level == "L3" and "untraced" in v.issues


def test_weak_value_mismatch_goes_to_l3_and_labels_have_no_containment():
    v = verdict([GET, archive(title="Q3 Balance Report")])          # 短标签:包含关系不算一致
    assert v.level == "L3" and "weak_mismatch" in v.issues


def test_numbers_inside_tolerance_pass_outside_go_to_l3():
    assert verdict([GET, archive(sheet={"sheet_id": "SHEET-0001", "amount": 28785.001})]).outcome == "pass"
    v = verdict([GET, archive(sheet={"sheet_id": "SHEET-0001", "amount": 28800})])
    assert v.level == "L3" and "weak_mismatch" in v.issues


def test_missing_action_fails_and_extra_action_goes_to_l3():
    assert verdict([GET]).reason == "missing_action"
    v = verdict([GET, archive(), ("create_note", {"title": "x"}, {"status": "saved"}, False)])
    assert v.level == "L3" and v.issues == ("extra_action",)


def test_empty_container_allowed_when_reference_is_empty():
    ref = task_ref_from_p0({"valid_paths": [{"execution_trace": [
        {"tool_name": "archive_sheet", "arguments": {"balance_sheet": {}, "title": "Q3 Balance"}, "output": {}}]}]})
    t = episode([archive(sheet={})])
    assert RU.judge(t, ref, CATALOG).reason != "empty_argument"
