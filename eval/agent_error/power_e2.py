"""
E2 配对比较(精确 McNemar)的最小可检测差异(MDE),用于给 ②、③ 定样本量。

设定:n 个配对,不一致配对(只有一组完成)占比 ψ = p_胜 + p_负,配对差 δ = p_胜 − p_负。
不一致配对数 D ~ Binomial(n, ψ);给定 D,胜的个数 ~ Binomial(D, (ψ + δ) / (2ψ))。
精确功效 = Σ_D P(D) · P(精确 McNemar 双侧 p ≤ α | D);MDE = 功效 ≥ 目标值的最小 δ(δ 最大为 ψ)。
近似公式(Connor 1987):n = (z_{α/2}·√ψ + z_β·√(ψ − δ²))² / δ²,对 δ 求解,作对照。

用法(仓库根目录):python eval/agent_error/power_e2.py --psi 0.15 --n 80 150 200
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from functools import lru_cache
from statistics import NormalDist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fc"))
from summarize_fc import mcnemar_exact  # noqa: E402

GRID = 0.0005          # δ 的搜索步长(0.05 个百分点)


def binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


@lru_cache(maxsize=None)
def rejection_wins(d: int, alpha: float) -> tuple[int, ...]:
    """D 个不一致配对时,双侧精确 McNemar 拒绝原假设的"胜"的个数。"""
    return tuple(w for w in range(d + 1) if mcnemar_exact(w, d - w) <= alpha)


def power_exact(n: int, psi: float, delta: float, alpha: float = 0.05) -> float:
    if not 0 < psi <= 1 or not 0 <= delta <= psi:
        raise ValueError(f"需要 0 < ψ ≤ 1 且 0 ≤ δ ≤ ψ,收到 ψ={psi}, δ={delta}")
    pi = (psi + delta) / (2 * psi)
    return sum(binom_pmf(d, n, psi) * sum(binom_pmf(w, d, pi) for w in rejection_wins(d, alpha))
               for d in range(n + 1))


def mde_exact(n: int, psi: float, power: float = 0.8, alpha: float = 0.05) -> float | None:
    """功效达到 power 的最小 δ(按 GRID 搜索);δ = ψ 时仍达不到则返回 None。"""
    steps = int(round(psi / GRID))
    for i in range(1, steps + 1):
        delta = i * GRID
        if power_exact(n, psi, delta, alpha) >= power:
            return delta
    return None


def mde_normal(n: int, psi: float, power: float = 0.8, alpha: float = 0.05) -> float | None:
    """Connor (1987) 近似公式对 δ 二分求解。"""
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)

    def needed(delta: float) -> float:
        return (za * math.sqrt(psi) + zb * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2

    lo, hi = 1e-9, psi
    if needed(hi) > n:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if needed(mid) > n else (lo, mid)
    return hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--psi", type=float, default=0.15, help="不一致配对占比 ψ")
    ap.add_argument("--n", type=int, nargs="+", default=[80, 150, 200])
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    print(f"ψ = {a.psi},α = {a.alpha}(双侧),功效 {a.power:.0%}")
    print("| n | MDE(精确 McNemar) | MDE(Connor 近似) |\n|---|---|---|")
    for n in a.n:
        ex, nm = mde_exact(n, a.psi, a.power, a.alpha), mde_normal(n, a.psi, a.power, a.alpha)
        fmt = lambda x: "达不到" if x is None else f"{x:.1%}"  # noqa: E731
        print(f"| {n} | {fmt(ex)} | {fmt(nm)} |")


if __name__ == "__main__":
    main()
