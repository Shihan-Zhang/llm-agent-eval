"""
Outcome Monitors(arXiv 2608.19303)的契约挖掘:从无扰动(P0)轨迹为每个工具学"结果契约",5 折按任务交叉拟合。

论文算法 2 的接受规则(样本数门槛照搬原文):
- 必有字段 F_req = 该工具所有训练结果字段集合的交集;全部字段 F_all = 并集;每个字段观察到的类型集合。
- echo:     参数名与结果字段同名,在 >= 2 对调用/结果中出现且全部相等。
- positivity: f ∈ F_req,在 >= 3 个样本中是数值且全部 > 0。
- domain:   f ∈ F_req,字段名命中分类词表,>= 4 个字符串样本,规范化后不同值 <= 8 个。
- affine:   (f, g) ∈ F_req²,>= 4 个样本、各自 >= 3 个不同值,拟合 y[g] = α·y[f] + β(挖掘容差 1e-8,运行时 1e-5)。
- order:    只在三组固定日期字段对中,>= 2 个可解析样本且全部 f <= g。
不变量必须在每个训练样本上成立。运行时触发的 8 类违反(论文附录 N):
missing_learned_field、unexpected_field、explicit_error、learned_type_mismatch、
learned_echo_mismatch、nonpositive_value、unseen_category、affine_relation_broken;
order 的违反记为 temporal_order_violated(论文未给代码名,见 README)。

论文没有公开的部分(分类词表、三组日期对)和我们自定的门槛都写在 SPEC 里,SPEC 的哈希记录在
eval/agent_error/README.md,测试保证两者一致——定下后不改。

信息边界:只读 P0 数据;不读任务描述、扰动字段、判分结果。运行时 check() 只看(工具名, 参数, 结果)。

用法(仓库根目录):
    python eval/agent_error/contracts.py build --source gold   # 题目自带的 C1–C4 标准 P0 轨迹(主分析)
    python eval/agent_error/contracts.py build --source e1     # E1 本地 4B 的 C1 P0 轨迹(敏感性分析)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TM = os.environ.get("TOOLMAZE_DIR", os.path.join(ROOT, "external", "agent_error", "toolmaze"))
E1_INFERENCES = os.path.join(ROOT, "external", "agent_error", "results", "e1", "qwen3i-8k", "fc", "c1", "inferences")
OUT = os.path.join(ROOT, "external", "agent_error", "results", "e2", "contracts")

# ---- 冻结的规格:改动任何一项都会让 test_spec_hash_matches_readme 失败 ----
SPEC: dict[str, Any] = {
    "lexicon": sorted([
        "action", "availability", "category", "channel", "class", "condition", "currency", "department",
        "direction", "exchange", "kind", "level", "method", "mode", "priority", "rating", "severity",
        "state", "status", "strategy", "tier", "trend", "type", "unit",
    ]),
    "date_pairs": [["check_in_date", "check_out_date"], ["start_time", "end_time"],
                   ["meeting_start_time", "meeting_end_time"]],
    "error_fields": ["error", "error_message", "errors", "exception"],
    "min_support": {"tool": 2, "type": 2, "echo": 2, "positivity": 3, "domain_samples": 4,
                    "domain_max_values": 8, "affine_samples": 4, "affine_distinct": 3, "order": 2},
    "tolerance": {"affine_fit": 1e-8, "affine_check": 1e-5},
    "folds": 5,
    "fold_seed": 20261108,
}
SPEC_SHA256 = hashlib.sha256(json.dumps(SPEC, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

MS = SPEC["min_support"]
TOL = SPEC["tolerance"]
LEXICON = frozenset(SPEC["lexicon"])
DATE_PAIRS = tuple(tuple(p) for p in SPEC["date_pairs"])
ERROR_FIELDS = frozenset(SPEC["error_fields"])


@dataclass(frozen=True)
class Sample:
    """一次成功的工具调用:workflow 是任务编号(如 C3_task_001),用于分折。"""
    workflow: str
    tool: str
    args: dict
    result: dict


@dataclass(frozen=True)
class Violation:
    property: str
    path: str
    expected: Any
    observed: Any
    detail: str

    def to_dict(self) -> dict:
        return {"property": self.property, "path": self.path, "expected": self.expected,
                "observed": self.observed, "detail": self.detail}


@dataclass(frozen=True)
class ToolContract:
    n: int
    required: tuple[str, ...]
    fields: tuple[str, ...]
    types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    echo: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()
    domains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    affine: tuple[tuple[str, str, float, float], ...] = ()
    order: tuple[tuple[str, str], ...] = ()

    def count(self) -> dict[str, int]:
        return {"required": len(self.required), "type": len(self.types), "echo": len(self.echo),
                "positivity": len(self.positive), "domain": len(self.domains),
                "affine": len(self.affine), "order": len(self.order)}

    def to_json(self) -> dict:
        return {"n": self.n, "required": list(self.required), "fields": list(self.fields),
                "types": {k: list(v) for k, v in self.types.items()}, "echo": list(self.echo),
                "positive": list(self.positive), "domains": {k: list(v) for k, v in self.domains.items()},
                "affine": [list(a) for a in self.affine], "order": [list(o) for o in self.order]}

    @staticmethod
    def from_json(d: dict) -> "ToolContract":
        return ToolContract(n=d["n"], required=tuple(d["required"]), fields=tuple(d["fields"]),
                            types={k: tuple(v) for k, v in d["types"].items()}, echo=tuple(d["echo"]),
                            positive=tuple(d["positive"]), domains={k: tuple(v) for k, v in d["domains"].items()},
                            affine=tuple((a[0], a[1], float(a[2]), float(a[3])) for a in d["affine"]),
                            order=tuple((o[0], o[1]) for o in d["order"]))


Registry = dict[str, ToolContract]


# ---------------------------------------------------------------- 基础工具
def flatten(obj: dict, prefix: str = "") -> dict[str, Any]:
    """嵌套字典展平为点号路径;列表作为叶子。"""
    out: dict[str, Any] = {}
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten(v, path))
        else:
            out[path] = v
    return out


def kind(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def in_lexicon(path: str) -> bool:
    return leaf(path).rsplit("_", 1)[-1].lower() in LEXICON


def norm(v: str) -> str:
    return v.strip().lower()


def parse_temporal(v: Any) -> tuple[str, Any] | None:
    """ISO 日期/日期时间或 HH:MM;返回(种类, 值),只有同种类才比较。"""
    if not isinstance(v, str):
        return None
    s = v.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return "date", date.fromisoformat(s)
        if re.match(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", s):
            return "datetime", datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
        if re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", s):
            return "time", time.fromisoformat(s)
    except ValueError:
        return None
    return None


def kind_ok(observed: str, allowed: tuple[str, ...]) -> bool:
    return observed in allowed or (observed == "integer" and "number" in allowed)


def has_error_field(result: dict) -> bool:
    return any(leaf(p).lower() in ERROR_FIELDS and v not in (None, "", [], {}) for p, v in flatten(result).items())


# ---------------------------------------------------------------- 读入 P0 轨迹
def _gold_steps(task: dict) -> Iterable[dict]:
    yield from task.get("execution_trace") or []
    for p in task.get("valid_paths") or []:
        yield from p.get("execution_trace") or []


def load_gold(tm_dir: str = TM, tiers: Iterable[str] = ("c1", "c2", "c3", "c4")) -> tuple[list[Sample], dict[str, str]]:
    """题目自带的 P0 标准轨迹。返回(样本, task_id → template_id)。只读 *_P0.json。"""
    samples: list[Sample] = []
    templates: dict[str, str] = {}
    seen: set[tuple] = set()
    for tier in tiers:
        for f in sorted(glob.glob(os.path.join(tm_dir, "data", "perturbed_tasks", tier, "*_P0.json"))):
            with open(f, encoding="utf-8") as fh:
                task = json.load(fh)
            if task.get("perturbation_mode", "P0") != "P0":
                continue
            wf = task["task_id"]
            templates[wf] = task.get("template_id") or wf
            for st in _gold_steps(task):
                out, args = st.get("output"), st.get("arguments") or {}
                if not isinstance(out, dict) or st.get("status", "success") != "success" or has_error_field(out):
                    continue
                key = (wf, st["tool_name"], json.dumps(args, sort_keys=True), json.dumps(out, sort_keys=True))
                if key in seen:  # 多条备选路径共享的前缀步骤只算一次
                    continue
                seen.add(key)
                samples.append(Sample(wf, st["tool_name"], args, out))
    return samples, templates


def load_e1(inference_dir: str = E1_INFERENCES) -> list[Sample]:
    """E1 本地模型的 P0 轨迹:只取未被注入、无 error 字段的工具结果,按 call_id 配对参数。"""
    samples: list[Sample] = []
    for f in sorted(glob.glob(os.path.join(inference_dir, "*_P0_inference.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("mode") != "P0":
            continue
        calls = {m["tool_call"]["id"]: m["tool_call"] for m in d["messages"] if m.get("type") == "tool_call"}
        for m in d["messages"]:
            if m.get("role") != "tool" or not isinstance(m.get("content"), dict):
                continue
            if (m.get("metadata") or {}).get("perturbation_status") != "clean" or has_error_field(m["content"]):
                continue
            call = calls.get(m.get("call_id"))
            if call:
                samples.append(Sample(d["task_id"], m["name"], call.get("arguments") or {}, m["content"]))
    return samples


# ---------------------------------------------------------------- 分折
def assign_folds(task_ids: Iterable[str], k: int = SPEC["folds"], seed: int = SPEC["fold_seed"]) -> dict[str, int]:
    ids = sorted(set(task_ids))
    random.Random(seed).shuffle(ids)
    return {t: i % k for i, t in enumerate(ids)}


def check_templates_disjoint(folds: dict[str, int], templates: dict[str, str]) -> None:
    where: dict[str, set[int]] = defaultdict(set)
    for t, f in folds.items():
        where[templates.get(t, t)].add(f)
    bad = [tp for tp, fs in where.items() if len(fs) > 1]
    if bad:
        raise ValueError(f"模板跨折,任务不再互斥:{bad[:5]}")


# ---------------------------------------------------------------- 挖掘(算法 2)
def _learn_tool(samples: list[Sample]) -> ToolContract | None:
    if len(samples) < MS["tool"]:
        return None
    flats = [flatten(s.result) for s in samples]
    required = set.intersection(*(set(f) for f in flats))
    all_fields = set().union(*(set(f) for f in flats))

    obs_kinds: dict[str, set[str]] = defaultdict(set)
    obs_count: dict[str, int] = defaultdict(int)
    for f in flats:
        for p, v in f.items():
            obs_kinds[p].add(kind(v))
            obs_count[p] += 1
    types = {p: tuple(sorted(ks)) for p, ks in obs_kinds.items() if obs_count[p] >= MS["type"]}

    echo_pairs: dict[str, list[bool]] = defaultdict(list)
    for s, f in zip(samples, flats):
        for a, v in s.args.items():
            if a in f and not isinstance(v, (dict, list)):
                echo_pairs[a].append(f[a] == v)
    echo = tuple(sorted(a for a, ok in echo_pairs.items() if len(ok) >= MS["echo"] and all(ok)))

    positive, domains, numeric = [], {}, []
    for p in sorted(required):
        vals = [f[p] for f in flats]
        if all(is_number(v) for v in vals):
            numeric.append(p)
            if len(vals) >= MS["positivity"] and all(v > 0 for v in vals):
                positive.append(p)
        if in_lexicon(p) and all(isinstance(v, str) for v in vals) and len(vals) >= MS["domain_samples"]:
            distinct = sorted({norm(v) for v in vals})
            if len(distinct) <= MS["domain_max_values"]:
                domains[p] = tuple(distinct)

    affine = []
    for i, fx in enumerate(numeric):
        for fy in numeric[i + 1:]:
            fit = _fit_affine([f[fx] for f in flats], [f[fy] for f in flats])
            if fit:
                affine.append((fx, fy, fit[0], fit[1]))

    order = []
    for a, b in DATE_PAIRS:
        pa = [p for p in all_fields if leaf(p) == a]
        for p1 in pa:
            p2 = p1[: -len(a)] + b
            parsed = [(parse_temporal(f.get(p1)), parse_temporal(f.get(p2))) for f in flats if p1 in f and p2 in f]
            comparable = [(x[1], y[1]) for x, y in parsed if x and y and x[0] == y[0]]
            if len(comparable) >= MS["order"] and all(x <= y for x, y in comparable):
                order.append((p1, p2))

    return ToolContract(n=len(samples), required=tuple(sorted(required)), fields=tuple(sorted(all_fields)),
                        types=types, echo=echo, positive=tuple(positive), domains=domains,
                        affine=tuple(affine), order=tuple(sorted(order)))


def _fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) < MS["affine_samples"] or len(set(xs)) < MS["affine_distinct"] or len(set(ys)) < MS["affine_distinct"]:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    alpha = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    beta = my - alpha * mx
    if all(abs(y - (alpha * x + beta)) <= TOL["affine_fit"] * max(1.0, abs(y)) for x, y in zip(xs, ys)):
        return alpha, beta
    return None


def learn(samples: Iterable[Sample]) -> Registry:
    by_tool: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_tool[s.tool].append(s)
    reg = {}
    for tool in sorted(by_tool):
        c = _learn_tool(by_tool[tool])
        if c:
            reg[tool] = c
    return reg


def crossfit(samples: list[Sample], folds: dict[str, int], k: int = SPEC["folds"]) -> list[Registry]:
    """第 k 份注册表只用不在第 k 折的任务训练。样本所属任务必须在分折表里。"""
    missing = sorted({s.workflow for s in samples} - set(folds))
    if missing:
        raise ValueError(f"样本的任务不在分折表中:{missing[:5]}")
    return [learn(s for s in samples if folds[s.workflow] != i) for i in range(k)]


def registry_for(registries: list[Registry], folds: dict[str, int], task_id: str) -> Registry:
    """评测 task_id 时只能用它所在折的注册表 C_k(该注册表没见过这道题)。"""
    return registries[folds[task_id]]


# ---------------------------------------------------------------- 运行时检查
def check(contract: ToolContract | None, args: dict, result: Any) -> list[Violation]:
    """返回有序的违反列表;没有契约或结果不是字典时返回空列表。"""
    if contract is None or not isinstance(result, dict):
        return []
    f = flatten(result)
    v: list[Violation] = []
    for p in contract.required:
        if p not in f:
            v.append(Violation("missing_learned_field", p, "present", "missing", f"missing nominal field {p}"))
    known = set(contract.fields)
    for p in f:
        if p not in known:
            v.append(Violation("unexpected_field", p, "absent", kind(f[p]), f"field absent from nominal traces: {p}"))
    errs = [p for p in f if leaf(p).lower() in ERROR_FIELDS and p not in known and f[p] not in (None, "", [], {})]
    if errs:
        v.append(Violation("explicit_error", ",".join(errs), "no error field", "error field present",
                           f"error-bearing field(s): {', '.join(errs)}"))
    for p, allowed in contract.types.items():
        if p in f and not kind_ok(kind(f[p]), allowed):
            v.append(Violation("learned_type_mismatch", p, list(allowed), kind(f[p]), f"field {p} has unexpected type"))
    for a in contract.echo:
        if a in args and a in f and args[a] != f[a]:
            v.append(Violation("learned_echo_mismatch", a, args[a], f[a], f"field {a} differs from call argument"))
    for p in contract.positive:
        if is_number(f.get(p)) and f[p] <= 0:
            v.append(Violation("nonpositive_value", p, "> 0", f[p], f"nominally positive field {p} is not positive"))
    for p, values in contract.domains.items():
        if isinstance(f.get(p), str) and norm(f[p]) not in values:
            v.append(Violation("unseen_category", p, list(values), f[p], f"field {p} has unseen categorical value"))
    for fx, fy, a, b in contract.affine:
        x, y = f.get(fx), f.get(fy)
        if is_number(x) and is_number(y):
            exp = a * x + b
            if abs(y - exp) > TOL["affine_check"] * max(1.0, abs(exp)):
                v.append(Violation("affine_relation_broken", f"{fx},{fy}", round(exp, 6), y,
                                   f"fields {fx} and {fy} violate a nominal relation"))
    for p1, p2 in contract.order:
        x, y = parse_temporal(f.get(p1)), parse_temporal(f.get(p2))
        if x and y and x[0] == y[0] and x[1] > y[1]:
            v.append(Violation("temporal_order_violated", f"{p1},{p2}", f"{p1} <= {p2}", [f[p1], f[p2]],
                               f"{p1} is later than {p2}"))
    return v


# ---------------------------------------------------------------- 落盘
def save(out_dir: str, registries: list[Registry], folds: dict[str, int], info: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, reg in enumerate(registries):
        with open(os.path.join(out_dir, f"fold_{i}.json"), "w", encoding="utf-8") as fh:
            json.dump({t: c.to_json() for t, c in reg.items()}, fh, ensure_ascii=False, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "folds.json"), "w", encoding="utf-8") as fh:
        json.dump(folds, fh, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "build_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)


def load(out_dir: str) -> tuple[list[Registry], dict[str, int]]:
    with open(os.path.join(out_dir, "folds.json"), encoding="utf-8") as fh:
        folds = json.load(fh)
    regs = []
    for i in range(SPEC["folds"]):
        with open(os.path.join(out_dir, f"fold_{i}.json"), encoding="utf-8") as fh:
            regs.append({t: ToolContract.from_json(c) for t, c in json.load(fh).items()})
    return regs, folds


def summarize(registries: list[Registry]) -> dict:
    per_fold = []
    for reg in registries:
        tot: dict[str, int] = defaultdict(int)
        for c in reg.values():
            for k, n in c.count().items():
                tot[k] += n
        per_fold.append({"tools": len(reg), **tot})
    return {"per_fold": per_fold, "invariants_total": sum(sum(v for k, v in p.items() if k != "tools") for p in per_fold)}


def _git_head() -> str:
    try:
        with open(os.path.join(ROOT, ".git", "HEAD"), encoding="utf-8") as fh:
            ref = fh.read().strip()
        if ref.startswith("ref: "):
            with open(os.path.join(ROOT, ".git", *ref[5:].split("/")), encoding="utf-8") as fh:
                return fh.read().strip()[:7]
        return ref[:7]
    except OSError:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--source", choices=["gold", "e1"], default="gold")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gold, templates = load_gold()
    if not gold:
        sys.exit(f"没有找到 P0 题目文件:{TM}")
    folds = assign_folds(templates)          # 两种来源共用同一分折(按全部 400 道 P0 题)
    check_templates_disjoint(folds, templates)
    samples = gold if a.source == "gold" else load_e1()
    regs = crossfit(samples, folds)
    out = a.out or os.path.join(OUT, a.source)
    info = {"source": a.source, "samples": len(samples), "workflows": len({s.workflow for s in samples}),
            "spec_sha256": SPEC_SHA256, "repo_commit": _git_head(), **summarize(regs)}
    save(out, regs, folds, info)
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
