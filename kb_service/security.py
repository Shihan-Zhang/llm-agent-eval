"""API Key 校验与按 Key 的令牌桶限流(线程安全)。

不复用 eval/agent_error/ratelimit.py:那里是调用方的限速器,额度不够就 sleep 等到够,且同时管 token 和
请求两种预算、带估算后结算;服务端需要的是「立刻拒绝 + 告诉对方等多久」(429 + Retry-After),
而且每个 Key 一个桶。那个文件也不在包里(靠 sys.path 导入),服务依赖实验脚本目录不合适。
"""
from __future__ import annotations

import hmac
import threading
import time
from dataclasses import dataclass
from typing import Callable

SECONDS_PER_MINUTE = 60.0


def key_matches(given: str | None, keys: frozenset[str]) -> bool:
    """逐个用常数时间比较,不因匹配位置不同泄露时序信息。"""
    if not given:
        return False
    matched = False
    for k in keys:
        matched |= hmac.compare_digest(given.encode("utf-8"), k.encode("utf-8"))
    return matched


@dataclass
class _Bucket:
    tokens: float
    last: float


class RateLimiter:
    """每个 Key 一个令牌桶:容量 = rate_per_min(允许的突发),每秒补 rate_per_min / 60 个。"""

    def __init__(self, rate_per_min: int, clock: Callable[[], float] = time.monotonic):
        if rate_per_min < 1:
            raise ValueError(f"rate_per_min 必须 ≥ 1,收到 {rate_per_min}")
        self.capacity = float(rate_per_min)
        self.per_second = rate_per_min / SECONDS_PER_MINUTE
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    @property
    def bucket_count(self) -> int:
        """当前桶的个数(只应等于通过鉴权的 Key 数;无效 Key 不能建桶,否则可被刷爆内存)。"""
        with self._lock:
            return len(self._buckets)

    def try_acquire(self, key: str) -> float:
        """拿到一个令牌返回 0;拿不到返回还要等的秒数(> 0),不扣令牌。"""
        with self._lock:
            now = self.clock()
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(self.capacity, now)
            b.tokens = min(self.capacity, b.tokens + max(0.0, now - b.last) * self.per_second)
            b.last = now
            if b.tokens >= 1:
                b.tokens -= 1
                return 0.0
            return (1 - b.tokens) / self.per_second
