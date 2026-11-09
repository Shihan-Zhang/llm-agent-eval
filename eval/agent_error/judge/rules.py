"""L1 / L2 规则判分。结果正确 = 完成必需动作 + 每个调用参数都能溯源到用户提示或未注入的工具输出。

L1:最终回答含损坏值 → 失败。没有最终回答不直接判失败,照常走 L2。
L2(按顺序):
  1. 任何调用(含失败的调用)的参数含损坏值 → 失败。
  1b. 任何成功执行的 Action 把 required 容器参数传空 → 失败(见 _empty_action_argument)。
  2. 逐条有效路径(P0 参考)检查,得到该路径的问题集合:
     - missing_action:某个 Action 工具既没有成功调用,也没有被其替代工具完成;
     - substitute:Action 由替代工具完成(没有参考参数可比);
     - extra_action:成功调用了路径外、也不是路径内动作替代品的 Action;
     - empty_argument:Action 的 required 容器参数在参考里非空、模型传空;替代工具没有参考,
       它的 required 容器参数传空({} 或 [])也算;
     - untraced:证据类值(ID、路径)在使用之前的用户提示与未注入输出里都找不到;
       长字符串找不到原文视为模型自己写的组合文本,只受第 1 条约束;
     - weak_mismatch / weak_unverified:required 参数里的弱值(日期、数字、小整数、布尔、短标签)
       与参考同一位置不一致 / 参考在该位置没有值;
     - reference_mismatch:参考的 required 原子值(长字符串除外)在模型参数里找不到。
     非 required 参数免于比对;包含关系只对 ID、日期、路径成立(normalize.CONTAINMENT_KINDS)。
  3. 没有路径动作齐全 → 失败(missing_action);有路径没有任何问题 → 通过;
     动作齐全的路径都有 empty_argument → 失败;其余 → L3(交给 LLM 裁判)。
"""
from __future__ import annotations

from dataclasses import dataclass

from .normalize import CONTAINMENT_KINDS, EVIDENCE_KINDS, canon, contained, flat, kind, leaves, norm, same, top_param
from .reference import Catalog, Step, TaskRef
from .trajectory import Call, Trajectory, contains_corrupted, corrupted_values

ISSUE_NAMES = {
    "missing_action": "缺必需动作", "substitute": "替代工具完成动作", "extra_action": "多余动作",
    "empty_argument": "必需参数为空", "untraced": "证据类值溯源不到", "weak_mismatch": "弱值与参考不一致",
    "weak_unverified": "弱值无参考可比", "reference_mismatch": "参考值在参数里找不到",
}


@dataclass(frozen=True)
class Verdict:
    level: str                      # "L1" / "L2" / "L3"
    outcome: str | None             # "fail" / "pass";L3 为 None
    reason: str                     # answer_corrupted / corrupted_argument / missing_action / empty_argument / pass / l3
    issues: tuple[str, ...] = ()    # L3 时最好那条路径的问题


def _empty(v: object) -> bool:
    return v in (None, "", {}, [])


def _sources(traj: Trajectory, before: int) -> tuple[list[tuple], str]:
    """调用 before 之前可用的来源:未注入的工具输出(规范化叶子与文本);用户提示另算。"""
    vals, text = [], []
    for c in traj.calls[:before]:
        if c.injected or c.output is None:
            continue
        vals += [canon(v) for _, v in leaves(c.output) if v is not None]
        text.append(norm(str(c.output)))
    return vals, "".join(text)


def _traced(key: str | None, value: object, traj: Trajectory, call: Call) -> bool:
    vals, text = _sources(traj, call.index)
    n = norm(value)
    if n and (n in norm(traj.prompt) or n in text):
        return True
    return any(same(canon(value), v) for v in vals)


def _matches(key: str | None, ref_value: object, model_values: list[object]) -> bool:
    c = canon(ref_value)
    if any(same(c, canon(m)) for m in model_values):
        return True
    return kind(key, ref_value) in CONTAINMENT_KINDS and any(contained(ref_value, m) for m in model_values if m is not None)


def _call_issues(call: Call, ref: Step, traj: Trajectory, catalog: Catalog) -> set[str]:
    issues = set()
    tool = call.tool
    if catalog.is_action(tool):
        for k, rv in ref.arguments.items():
            if catalog.is_required(tool, k) and isinstance(rv, (dict, list)) and rv and _empty(call.arguments.get(k)):
                issues.add("empty_argument")
    ref_flat = dict(flat(ref.arguments))
    model_flat = list(flat(call.arguments))
    for path, v in model_flat:
        if v is None:
            continue
        key = path.rsplit("/", 1)[-1].removesuffix("[]")
        k = kind(key, v)
        if k in EVIDENCE_KINDS:
            if not _traced(key, v, traj, call) and k != "long":
                issues.add("untraced")
            continue
        if not catalog.is_required(tool, top_param(path)):
            continue
        if path not in ref_flat or ref_flat[path] is None:
            issues.add("weak_unverified")
        elif not _matches(key, ref_flat[path], [v]):
            issues.add("weak_mismatch")
    model_values = [v for _, v in model_flat]
    for path, rv in ref_flat.items():
        if rv is None or not catalog.is_required(tool, top_param(path)):
            continue
        rkey = path.rsplit("/", 1)[-1].removesuffix("[]")
        if isinstance(rv, str) and kind(rkey, rv) == "long":
            continue
        if not _matches(rkey, rv, model_values):
            issues.add("reference_mismatch")
    return issues


def _path_issues(path: tuple[Step, ...], traj: Trajectory, catalog: Catalog) -> set[str]:
    ok_calls = [c for c in traj.calls if c.ok]
    done = {c.tool for c in ok_calls}
    actions = [s for s in path if catalog.is_action(s.tool)]
    issues = set()
    for s in actions:
        if s.tool in done:
            continue
        if catalog.substitutes.get(s.tool, frozenset()) & done:
            issues.add("substitute")
        else:
            return {"missing_action"}
    path_tools = {s.tool for s in path}
    action_subs = set().union(*(catalog.substitutes.get(s.tool, frozenset()) for s in actions)) if actions else set()
    for c in ok_calls:
        if c.tool not in path_tools:
            if catalog.is_action(c.tool) and c.tool not in action_subs:
                issues.add("extra_action")
            continue
        candidates = [_call_issues(c, s, traj, catalog) for s in path if s.tool == c.tool]
        issues |= min(candidates, key=len)
    return issues


def _empty_action_argument(traj: Trajectory, ref: TaskRef, catalog: Catalog) -> bool:
    """任何成功执行的 Action(不论在哪条路径上、是不是替代工具),required 容器参数传空({} 或 [])
    就是带着缺失数据执行了动作。参考里同一工具、同一参数本来就为空的除外
    (ToolMaze 全部 P0 参考的 3,664 次 Action 调用里只有 1 次如此)。"""
    allowed = {(s.tool, k) for p in ref.paths for s in p for k, v in s.arguments.items()
               if isinstance(v, (dict, list)) and not v}
    return any(catalog.is_action(c.tool) and catalog.is_required(c.tool, k) and isinstance(v, (dict, list)) and not v
               and (c.tool, k) not in allowed
               for c in traj.calls if c.ok for k, v in c.arguments.items())


def judge(traj: Trajectory, ref: TaskRef, catalog: Catalog) -> Verdict:
    corrupted = corrupted_values(traj, ref)
    if traj.final_answer and contains_corrupted(traj.final_answer, corrupted):
        return Verdict("L1", "fail", "answer_corrupted")
    for c in traj.calls:
        if any(v is not None and not isinstance(v, bool) and contains_corrupted(v, corrupted)
               for _, v in leaves(c.arguments)):
            return Verdict("L2", "fail", "corrupted_argument")
    if _empty_action_argument(traj, ref, catalog):
        return Verdict("L2", "fail", "empty_argument")
    per_path = [_path_issues(p, traj, catalog) for p in ref.paths]
    complete = [i for i in per_path if "missing_action" not in i]
    if not complete:
        return Verdict("L2", "fail", "missing_action")
    if any(not i for i in complete):
        return Verdict("L2", "pass", "pass")
    if all("empty_argument" in i for i in complete):
        return Verdict("L2", "fail", "empty_argument")
    best = min(complete, key=len)
    return Verdict("L3", None, "l3", tuple(sorted(best)))
