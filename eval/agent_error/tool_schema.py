"""ToolMaze 工具 schema 里「只写 type: object、没有 properties」的参数(下称无属性对象)。

ToolMaze 有 111 个顶层参数和 8 个数组元素是无属性对象。① 的 API 运行里,这类参数 793/793 次被传成 {};
本地 Ollama 运行 186/186 次是非空对象(eval/results/e2_judge_survey.md 第 6 节)。
schema_probe/ 用这里的变换做最小复现:B = 补 additionalProperties: true,C = 按参考参数补 properties。
所有函数都返回新对象,不改传入的 schema。
"""
from __future__ import annotations

import copy
import os
from collections import Counter
from typing import Callable

Schema = dict


def is_bare_object(schema: object) -> bool:
    return isinstance(schema, dict) and schema.get("type") == "object" and not schema.get("properties")


def bare_object_paths(parameters: Schema, prefix: str = "") -> list[str]:
    """parameters(一个工具的 JSON Schema)里所有无属性对象的位置,形如 /contact_card、/items[]。"""
    out = []
    for name, sub in (parameters.get("properties") or {}).items():
        out += _walk(sub, f"{prefix}/{name}")
    return out


def _walk(schema: object, path: str) -> list[str]:
    if not isinstance(schema, dict):
        return []
    if is_bare_object(schema):
        return [path]
    out = []
    for name, sub in (schema.get("properties") or {}).items():
        out += _walk(sub, f"{path}/{name}")
    if isinstance(schema.get("items"), dict):
        out += _walk(schema["items"], f"{path}[]")
    return out


def load_bare_params(definitions_dir: str) -> frozenset[tuple[str, str]]:
    """ToolMaze 工具定义里所有顶层无属性对象参数 {(工具名, 参数名)}。"""
    import yaml
    out = set()
    for fn in ("source.yaml", "action.yaml", "processor.yaml"):
        with open(os.path.join(definitions_dir, fn), encoding="utf-8") as fh:
            for t in yaml.safe_load(fh)["tools"]:
                params = (((t.get("paradigms") or {}).get("function_call") or {}).get("spec") or {}).get("parameters") or {}
                out |= {(t["name"], p[1:]) for p in bare_object_paths(params) if p.count("/") == 1 and not p.endswith("[]")}
    return frozenset(out)


def bare_argument_states(messages: list[dict], bare: frozenset[tuple[str, str]]) -> Counter:
    """一个推理文件(*_inference.json 的 messages)里,顶层无属性对象参数的取值:empty({}、""、null)/ nonempty(非空对象)/ other。"""
    c = Counter()
    for m in messages:
        tc = m.get("tool_call")
        if not tc:
            continue
        for k, v in (tc.get("arguments") or {}).items():
            if (tc.get("name"), k) in bare:
                c["empty" if v in ({}, "", None) else "nonempty" if isinstance(v, dict) else "other"] += 1
    return c


def map_bare_objects(parameters: Schema, fn: Callable[[str, Schema], Schema]) -> Schema:
    """返回 parameters 的副本,其中每个无属性对象 s(位置 path)换成 fn(path, s)。fn 不得修改 s。"""
    def rebuild(schema: object, path: str) -> object:
        if not isinstance(schema, dict):
            return copy.deepcopy(schema)
        if is_bare_object(schema):
            return fn(path, copy.deepcopy(schema))
        new = {k: copy.deepcopy(v) for k, v in schema.items() if k not in ("properties", "items")}
        if "properties" in schema:
            new["properties"] = {k: rebuild(v, f"{path}/{k}") for k, v in schema["properties"].items()}
        if "items" in schema:
            new["items"] = rebuild(schema["items"], f"{path}[]")
        return new

    new = {k: copy.deepcopy(v) for k, v in parameters.items() if k != "properties"}
    if "properties" in parameters:
        new["properties"] = {k: rebuild(v, f"/{k}") for k, v in parameters["properties"].items()}
    return new


def allow_additional_properties(parameters: Schema) -> Schema:
    """变体 B:无属性对象加 additionalProperties: true(已显式写了的不动)。"""
    return map_bare_objects(parameters, lambda _p, s: s if "additionalProperties" in s else {**s, "additionalProperties": True})


def map_tools(tools: list[dict], fn: Callable[[Schema], Schema]) -> list[dict]:
    """OpenAI 格式的工具列表 [{type, function: {name, description, parameters}}] → 新列表,parameters 换成 fn(parameters)。"""
    out = []
    for t in tools:
        f = t.get("function") or {}
        params = f.get("parameters")
        new_f = {**copy.deepcopy(f), "parameters": fn(params)} if isinstance(params, dict) else copy.deepcopy(f)
        out.append({**copy.deepcopy(t), "function": new_f})
    return out


def json_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def schema_from_example(value: object) -> Schema:
    """按一个示例值推出 JSON Schema(变体 C 用;只做诊断,不打算采用)。"""
    t = json_type(value)
    if t == "object":
        return {"type": "object", "properties": {k: schema_from_example(v) for k, v in value.items()}}
    if t == "array":
        return {"type": "array", "items": schema_from_example(value[0]) if value else {}}
    return {"type": t} if t != "null" else {}
