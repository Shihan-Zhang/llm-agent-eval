"""
按 token 和请求数限速的令牌桶(硅基流动 L0:每分钟 40,000 token、1,000 次请求),线程安全。

每次请求前先估算这次要用的 token:输入 = 消息与工具定义的字符数 ÷ 每 token 字符数(用已完成请求的
实际 prompt_tokens 在线校准,初值保守),输出 = max_tokens × 经验比例(由试跑实测)。额度不够就等;
请求完成后用实际用量结算(多退少补,桶可以短暂为负,之后的请求相应多等)。重试作为兜底,不在这里。
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Limits:
    tpm: int = 40_000
    rpm: int = 1_000
    headroom: float = 0.9          # 只用限额的 90%,给估算误差和服务端计时差留余量


class TokenBucket:
    def __init__(self, limits: Limits = Limits(), clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.cap_tok = limits.tpm * limits.headroom
        self.cap_req = limits.rpm * limits.headroom
        self.rate_tok = self.cap_tok / 60.0
        self.rate_req = self.cap_req / 60.0
        self.tok = self.cap_tok
        self.req = self.cap_req
        self.clock, self.sleep = clock, sleep
        self.last = clock()
        self.waited_s = 0.0
        self.lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock()
        dt = max(0.0, now - self.last)
        self.last = now
        self.tok = min(self.cap_tok, self.tok + dt * self.rate_tok)
        self.req = min(self.cap_req, self.req + dt * self.rate_req)

    def acquire(self, tokens: float) -> float:
        """扣除一次请求和估算的 token;额度不够就睡到够为止。返回本次等待秒数。"""
        need = min(tokens, self.cap_tok)          # 单次估算超过容量时,至多等到桶满
        waited = 0.0
        while True:
            with self.lock:
                self._refill()
                if self.tok >= need and self.req >= 1:
                    self.tok -= tokens
                    self.req -= 1
                    self.waited_s += waited
                    return waited
                wait = max((need - self.tok) / self.rate_tok, (1 - self.req) / self.rate_req, 0.01)
            self.sleep(wait)
            waited += wait

    def settle(self, estimated: float, actual: float) -> None:
        """用实际用量修正:估多了退回,估少了补扣。"""
        with self.lock:
            self.tok = min(self.cap_tok, self.tok + estimated - actual)


class TokenEstimator:
    """输入 token ≈ 字符数 / chars_per_token(用实际 prompt_tokens 在线校准);输出 ≈ max_tokens × output_ratio。"""

    def __init__(self, chars_per_token: float = 2.5, output_ratio: float = 0.05):
        self.cpt0 = chars_per_token
        self.output_ratio = output_ratio
        self.chars = 0
        self.prompt_tokens = 0
        self.lock = threading.Lock()

    @property
    def cpt(self) -> float:
        with self.lock:
            if self.prompt_tokens < 2_000:            # 样本太少时用保守初值
                return self.cpt0
            return min(self.cpt0 * 2, self.chars / self.prompt_tokens)

    def estimate(self, chars: int, max_tokens: int) -> int:
        return math.ceil(chars / self.cpt) + math.ceil(max_tokens * self.output_ratio)

    def observe(self, chars: int, prompt_tokens: int | None) -> None:
        if prompt_tokens:
            with self.lock:
                self.chars += chars
                self.prompt_tokens += prompt_tokens
