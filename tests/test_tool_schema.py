"""eval/agent_error/tool_schema.py:找出无属性对象参数、补 additionalProperties、按示例推 schema。不读 external/。"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "agent_error"))
import tool_schema as TS  # noqa: E402

PARAMS = {
    "type": "object",
    "properties": {
        "alert_card": {"type": "object", "description": "WeatherAPI-native alert card object"},
        "title": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object"}},
        "typed": {"type": "object", "properties": {"id": {"type": "string"}, "meta": {"type": "object"}}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "explicit": {"type": "object", "additionalProperties": False},
    },
    "required": ["alert_card", "title"],
}


def test_bare_object_paths_cover_top_level_nested_and_array_items():
    assert TS.bare_object_paths(PARAMS) == ["/alert_card", "/items[]", "/typed/meta", "/explicit"]


def test_allow_additional_properties_only_touches_bare_objects_and_keeps_input():
    before = copy.deepcopy(PARAMS)
    out = TS.allow_additional_properties(PARAMS)
    assert PARAMS == before                                            # 不改传入对象
    assert out["properties"]["alert_card"] == {"type": "object", "description": "WeatherAPI-native alert card object",
                                               "additionalProperties": True}
    assert out["properties"]["items"]["items"] == {"type": "object", "additionalProperties": True}
    assert out["properties"]["typed"]["properties"]["meta"]["additionalProperties"] is True
    assert out["properties"]["explicit"] == {"type": "object", "additionalProperties": False}   # 显式写了的不动
    assert out["properties"]["title"] == {"type": "string"} and out["required"] == ["alert_card", "title"]
    assert TS.bare_object_paths(out) == TS.bare_object_paths(PARAMS)   # 仍然没有 properties


def test_map_tools_returns_new_list_in_openai_format():
    tools = [{"type": "function", "function": {"name": "archive", "parameters": PARAMS}},
             {"type": "function", "function": {"name": "noparams"}}]
    before = copy.deepcopy(tools)
    out = TS.map_tools(tools, TS.allow_additional_properties)
    assert tools == before
    assert out[0]["function"]["parameters"]["properties"]["alert_card"]["additionalProperties"] is True
    assert out[1] == tools[1]


def test_bare_argument_states_counts_empty_nonempty_other():
    bare = frozenset({("archive", "alert_card")})
    msgs = [{"role": "user", "content": "x"},
            {"role": "assistant", "tool_call": {"name": "archive", "arguments": {"alert_card": {}, "title": "t"}}},
            {"role": "assistant", "tool_call": {"name": "archive", "arguments": {"alert_card": {"id": "A1"}}}},
            {"role": "assistant", "tool_call": {"name": "archive", "arguments": {"alert_card": ""}}},
            {"role": "assistant", "tool_call": {"name": "archive", "arguments": {"alert_card": "A1"}}},
            {"role": "assistant", "tool_call": {"name": "other", "arguments": {"alert_card": {}}}}]
    assert TS.bare_argument_states(msgs, bare) == {"empty": 2, "nonempty": 1, "other": 1}


def test_load_bare_params_reads_top_level_bare_objects(tmp_path):
    spec = {"name": "archive", "paradigms": {"function_call": {"spec": {"name": "archive", "parameters": PARAMS}}}}
    for fn in ("source.yaml", "action.yaml", "processor.yaml"):
        (tmp_path / fn).write_text(json.dumps({"tools": [spec] if fn == "action.yaml" else []}), encoding="utf-8")
    assert TS.load_bare_params(str(tmp_path)) == {("archive", "alert_card"), ("archive", "explicit")}


def test_schema_from_example():
    ex = {"alert_card_id": "wapi_new_york", "celsius_now": 15, "ok": True, "rate": 1.5, "tags": ["a"], "none": None}
    assert TS.schema_from_example(ex) == {"type": "object", "properties": {
        "alert_card_id": {"type": "string"}, "celsius_now": {"type": "integer"}, "ok": {"type": "boolean"},
        "rate": {"type": "number"}, "tags": {"type": "array", "items": {"type": "string"}}, "none": {}}}
