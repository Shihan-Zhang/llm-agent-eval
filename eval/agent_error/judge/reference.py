"""参考数据:工具目录(类别、required 参数、替代工具)与每道题 P0 的有效路径。

P1–P4 文件里的 expected_result 带着注入后的坏数据(is_perturbed),不能当正确答案;
正确值一律取同一题的 P0 文件(C1 是 expected_result.tool_calls,C2–C4 是 valid_paths[].execution_trace)。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import yaml

from .normalize import kind, leaves, norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import receipt as R  # noqa: E402

DEFINITION_FILES = ("source.yaml", "action.yaml", "processor.yaml")


@dataclass(frozen=True)
class Catalog:
    category: dict[str, str]                       # 工具 → Source / Action / Processor
    required: dict[str, frozenset[str]]            # 工具 → required 参数名
    substitutes: dict[str, frozenset[str]]         # 工具 → 可替代它的工具(substitutes 与 alternatives.yaml)

    def is_action(self, tool: str) -> bool:
        return self.category.get(tool) == "Action"

    def is_required(self, tool: str, param: str) -> bool:
        return param in self.required.get(tool, frozenset())


def load_catalog(definitions_dir: str) -> Catalog:
    category, required = {}, {}
    for name in DEFINITION_FILES:
        with open(os.path.join(definitions_dir, name), encoding="utf-8") as fh:
            for tool in yaml.safe_load(fh).get("tools") or []:
                category[tool["name"]] = tool.get("category")
                spec = (((tool.get("paradigms") or {}).get("function_call") or {}).get("spec") or {})
                required[tool["name"]] = frozenset((spec.get("parameters") or {}).get("required") or [])
    subs = {k: frozenset(v) for k, v in R.build_subs(definitions_dir).items()}
    return Catalog(category, required, subs)


@dataclass(frozen=True)
class Step:
    tool: str
    arguments: dict
    output: object


@dataclass(frozen=True)
class TaskRef:
    paths: tuple[tuple[Step, ...], ...]            # 每条有效路径的参考步骤
    clean_values: frozenset[str] = field(default_factory=frozenset)   # 所有参考输出叶子的规范化值

    def last_step_ids(self) -> list[set[str]]:
        """每条路径最后一步输出里的 ID 类值(去掉空值与 N/A),用于只看最终回答的 A/B/C/D 调查。"""
        out = []
        for path in self.paths:
            out.append({str(v) for k, v in leaves(path[-1].output)
                        if v not in (None, "", "N/A") and kind(k, v) == "id" and len(norm(v)) >= 4})
        return out


def _steps(trace: list[dict]) -> tuple[Step, ...]:
    return tuple(Step(s["tool_name"], s.get("arguments") or {}, s.get("output")) for s in trace)


def task_ref_from_p0(p0: dict) -> TaskRef:
    if "valid_paths" in p0:
        paths = tuple(_steps(vp["execution_trace"]) for vp in p0["valid_paths"])
    else:
        paths = (_steps(p0["expected_result"]["tool_calls"]),)
    clean = frozenset(norm(v) for path in paths for s in path for _, v in leaves(s.output) if v is not None)
    return TaskRef(paths, clean)


def load_task_ref(toolmaze_dir: str, tier: str, task_id: str) -> TaskRef:
    path = os.path.join(toolmaze_dir, "data", "perturbed_tasks", tier, f"{task_id}_P0.json")
    with open(path, encoding="utf-8") as fh:
        return task_ref_from_p0(json.load(fh))
