"""值的规范化与比较:数字按容差、日期转 ISO、字符串小写去标点;值的类别(证据类 / 弱值)。

阈值是命名常量,人工核对后再调(调整要写进 README 并重跑 survey.py)。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterator

ABS_TOL = 0.005            # 数字绝对容差
REL_TOL = 1e-4             # 数字相对容差
SMALL_INT = 100            # |n| < SMALL_INT 的整数是小整数(弱值)
LONG_WORDS = 4             # ≥ LONG_WORDS 个词的字符串是长字符串
MIN_MATCH = 4              # 子串匹配要求规范化后至少这么长(避免碰巧命中)

DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S",
                "%Y/%m/%d", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y", "%d %B %Y", "%d %b %Y")
TIME_FORMATS = ("%H:%M", "%H:%M:%S")
_NUM = re.compile(r"^-?\d+(\.\d+)?$")
_ID_KEY = re.compile(r"(^|_)(id|code|ref|number|no|sku|uuid|token)$", re.I)
_ID_LIKE = re.compile(r"^(?=.*\d)[A-Za-z0-9_\-]{4,}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]+$", re.I)
_PATH = re.compile(r"^(/|[A-Za-z]:\\|~/)?[\w.\-]+([/\\][\w.\-]+)+/?$")


def norm(value: Any) -> str:
    """小写,去掉空白、标点和下划线。"""
    return re.sub(r"[\W_]+", "", str(value).lower())


def leaves(obj: Any, key: str | None = None) -> Iterator[tuple[str | None, Any]]:
    """(最近的键名, 叶子值);空列表产出 (键, None)。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, k)
    elif isinstance(obj, list):
        if not obj:
            yield key, None
        for v in obj:
            yield from leaves(v, key)
    else:
        yield key, obj


def flat(obj: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """(参数路径, 叶子值),路径形如 /balance_sheet/sheet_id、/tags[]。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flat(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for v in obj:
            yield from flat(v, prefix + "[]")
    else:
        yield prefix, obj


def top_param(path: str) -> str:
    """参数路径的顶层参数名:/balance_sheet/sheet_id → balance_sheet。"""
    return path.lstrip("/").split("/", 1)[0].removesuffix("[]")


def as_iso(text: str) -> str | None:
    """日期 → YYYY-MM-DD[THH:MM:SS];只有时间 → THH:MM:SS;不是日期返回 None。"""
    s = text.strip()
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(s, f).isoformat()
        except ValueError:
            pass
    for f in TIME_FORMATS:
        try:
            return "T" + datetime.strptime(s, f).strftime("%H:%M:%S")
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None).isoformat()
    except ValueError:
        return None


def canon(value: Any) -> tuple[str, Any] | None:
    """('bool', b) / ('num', float) / ('date', iso) / ('str', 规范化串);None → None。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    s = str(value)
    if _NUM.match(s.strip()):
        return ("num", float(s))
    iso = as_iso(s)
    return ("date", iso) if iso else ("str", norm(s))


def same(a: tuple[str, Any] | None, b: tuple[str, Any] | None) -> bool:
    """规范化值相等:数字按容差;只有时间的日期与带日期的比时间部分。"""
    if a is None or b is None or a[0] != b[0]:
        return False
    if a[0] == "num":
        return abs(a[1] - b[1]) <= max(ABS_TOL, REL_TOL * max(abs(a[1]), abs(b[1])))
    if a[0] == "date" and (a[1].startswith("T") or b[1].startswith("T")):
        return a[1].endswith(b[1]) or b[1].endswith(a[1])
    return a[1] == b[1]


def kind(key: str | None, value: Any) -> str:
    """值的类别。证据类:'id'、'path'、'long';其余:'date'、'number'、'small_int'、'bool'、'label'。"""
    if isinstance(value, bool):
        return "bool"
    c = canon(value)
    if c is None:
        return "label"
    if c[0] == "num":
        is_int = float(c[1]).is_integer()
        if _ID_KEY.search(key or "") and not isinstance(value, float):
            return "id"
        return "small_int" if is_int and abs(c[1]) < SMALL_INT else "number"
    if c[0] == "date":
        return "date"
    s = str(value).strip()
    if len(s.split()) >= LONG_WORDS:
        return "long"
    if _EMAIL.match(s) or _ID_KEY.search(key or "") or _ID_LIKE.match(s):
        return "id"
    if " " not in s and _PATH.match(s):
        return "path"
    return "label"


EVIDENCE_KINDS = frozenset({"id", "path", "long"})          # 可用来证明溯源
CONTAINMENT_KINDS = frozenset({"id", "date", "path"})       # 与参考比对时允许包含关系(短标签不允许)


def contained(short: Any, long_value: Any) -> bool:
    """规范化后 short 是 long_value 的子串(长度 ≥ MIN_MATCH);日期先转 ISO 再看。"""
    c = canon(short)
    if c and c[0] == "date":
        text = str(long_value)
        iso_parts = {as_iso(m.group(0)) for m in re.finditer(
            r"\d{4}[-/]\d{2}[-/]\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}(?::\d{2})?", text)}
        return any(same(c, ("date", p)) for p in iso_parts if p)
    n = norm(short)
    return len(n) >= MIN_MATCH and n in norm(long_value)
