"""
Outcome Monitors(arXiv 2608.19303)的回执:工具结果违反契约时,在原结果外附一张非强制的回执(论文算法 1)。

- 没有违反:原样返回同一个结果对象,模型看到的与上游完全一致。
- 有违反:返回新字典 {"tool_result": 原结果, "outcome_contract": 回执};原结果不复制、不修改。
- 回执用论文 v2 格式(只写第一条违反的五元组:tool, property, path, expected, observed),
  admissible_recovery_tools = [出错的工具] + (Subs(τ) ∩ 本题可见的工具),所以不会列出模型调不到的工具。
- variant="stripped" 去掉 admissible_recovery_tools(论文附录 L 的对照组),其余逐字相同。

Subs(τ) 只来自两个公开、与任务无关的来源:工具定义里的 substitutes 字段,以及 alternatives.yaml 中
同组兄弟路径的入口工具。信息边界:augment() 只接收(工具名, 参数, 结果, 契约, 可见工具),不接触题目、
扰动标记、标准答案或判分结果。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import yaml

from contracts import ToolContract, Violation, check

SCHEMA_VERSION = "tool_property_path_expected_observed_first_v2"
VARIANTS = ("full", "stripped")
DEFINITION_FILES = ("source.yaml", "action.yaml", "processor.yaml")

Subs = dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Outcome:
    """model_view 是交给模型的内容;violations 是全部违反(只写日志,模型只看到第一条)。"""
    model_view: Any
    violations: tuple[Violation, ...]
    receipt: dict | None


def _read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_subs(definitions_dir: str) -> Subs:
    """从 substitutes 字段与 alternatives.yaml 的兄弟路径入口工具构造 Subs(τ)(不含 τ 自身,排序)。"""
    subs: dict[str, set[str]] = {}
    for name in DEFINITION_FILES:
        path = os.path.join(definitions_dir, name)
        if not os.path.exists(path):
            continue
        for tool in _read_yaml(path).get("tools") or []:
            subs.setdefault(tool["name"], set()).update(tool.get("substitutes") or [])
    alt_path = os.path.join(definitions_dir, "alternatives.yaml")
    if os.path.exists(alt_path):
        for groups in (_read_yaml(alt_path).get("alternatives") or {}).values():
            for group in (groups or {}).values():
                paths = [p.get("tools") or [] for p in group.get("paths") or []]
                entries = [p[0]["tool_name"] for p in paths if p]
                for i, tools in enumerate(paths):
                    siblings = {e for j, e in enumerate(entries) if j != i}
                    for t in tools:
                        subs.setdefault(t["tool_name"], set()).update(siblings)
    return {t: tuple(sorted(s - {t})) for t, s in sorted(subs.items())}


def recovery_tools(tool: str, subs: Subs, visible: Iterable[str]) -> list[str]:
    """R = {τ} ∪ {s ∈ Subs(τ) : s 可见};τ 在首位,其余按名字排序。"""
    vis = set(visible)
    return [tool] + [s for s in subs.get(tool, ()) if s in vis and s != tool]


def make_receipt(tool: str, first: Violation, recovery: list[str], variant: str) -> dict:
    receipt = {"status": "inconsistent",
               "violation": {"tool": tool, "property": first.property, "path": first.path,
                             "expected": first.expected, "observed": first.observed}}
    if variant == "full":
        receipt["admissible_recovery_tools"] = recovery
    return receipt


def augment(tool: str, args: dict, result: Any, contract: ToolContract | None, subs: Subs,
            visible: Iterable[str], variant: str = "full") -> Outcome:
    if variant not in VARIANTS:
        raise ValueError(f"variant 只能是 {VARIANTS},收到 {variant!r}")
    violations = tuple(check(contract, args or {}, result))
    if not violations:
        return Outcome(result, (), None)
    receipt = make_receipt(tool, violations[0], recovery_tools(tool, subs, visible), variant)
    return Outcome({"tool_result": result, "outcome_contract": receipt}, violations, receipt)
