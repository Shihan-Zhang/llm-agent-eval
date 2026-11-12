"""把 ToolMaze 推理文件(*_inference.json)解析成调用序列。

推理文件里的工具结果是原始结果(上游先记日志、再交给模型),不含回执;注入标签在
tool 消息的 metadata.perturbation_status(clean / perturbed)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .normalize import MIN_MATCH, leaves, norm
from .reference import TaskRef


@dataclass(frozen=True)
class Call:
    index: int                # 在调用序列中的位置(0 起)
    tool: str
    arguments: dict
    output: object            # 工具结果;没有对应结果时为 None
    injected: bool            # 这次结果是被注入的
    ok: bool                  # 未注入、有结果、结果不带非空 error 字段


@dataclass(frozen=True)
class Trajectory:
    prompt: str               # 第一条 user 消息
    calls: tuple[Call, ...]
    final_answer: str | None  # 最后一条消息是不带工具调用的助手消息时的内容;否则 None(多为轮数用尽)


def parse(messages: list[dict]) -> Trajectory:
    prompt = messages[0].get("content") or "" if messages and messages[0].get("role") == "user" else ""
    pending: dict[str, int] = {}
    raw: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_call"):
            tc = m["tool_call"]
            pending[tc.get("id")] = len(raw)
            raw.append({"tool": tc.get("name"), "arguments": tc.get("arguments") or {}, "output": None, "injected": False})
        elif m.get("role") == "tool" and m.get("call_id") in pending:
            r = raw[pending[m["call_id"]]]
            r["output"] = m.get("content")
            r["injected"] = (m.get("metadata") or {}).get("perturbation_status") == "perturbed"
    calls = []
    for i, r in enumerate(raw):
        out = r["output"]
        ok = (not r["injected"]) and isinstance(out, dict) and not out.get("error")
        calls.append(Call(i, r["tool"], r["arguments"] if isinstance(r["arguments"], dict) else {}, out, r["injected"], ok))
    last = messages[-1] if messages else {}
    final = None
    if last.get("role") == "assistant" and not last.get("tool_call") and last.get("type") != "tool_call":
        final = last.get("content") or ""
    return Trajectory(prompt, tuple(calls), final)


def load(path: str) -> Trajectory:
    with open(path, encoding="utf-8") as fh:
        return parse(json.load(fh)["messages"])


def corrupted_values(traj: Trajectory, ref: TaskRef) -> frozenset[str]:
    """注入结果里、规范化后长度 ≥ MIN_MATCH 且不出现在任何参考输出里的值(规范化形式)。"""
    out = set()
    for c in traj.calls:
        if c.injected:
            for _, v in leaves(c.output):
                if v is None or isinstance(v, bool):
                    continue
                n = norm(v)
                if len(n) >= MIN_MATCH and n not in ref.clean_values:
                    out.add(n)
    return frozenset(out)


def contains_corrupted(value: object, corrupted: frozenset[str]) -> bool:
    n = norm(value)
    return any(x == n or x in n for x in corrupted)
