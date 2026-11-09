import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "agent_error"))
import power_e2 as P  # noqa: E402


def test_power_is_alpha_bounded_at_null_and_monotone():
    assert P.power_exact(80, 0.15, 0.0) <= 0.05                      # 精确检验偏保守
    a, b, c = (P.power_exact(80, 0.15, d) for d in (0.05, 0.10, 0.15))
    assert a < b < c


def test_power_matches_brute_force_small_case():
    # n = 3:逐个枚举 (胜, 负, 平) 的多项分布
    n, psi, delta, alpha = 3, 0.6, 0.4, 0.5
    pw, pl = (psi + delta) / 2, (psi - delta) / 2
    from math import factorial
    total = 0.0
    for w in range(n + 1):
        for l in range(n + 1 - w):
            t = n - w - l
            prob = factorial(n) / (factorial(w) * factorial(l) * factorial(t)) * pw ** w * pl ** l * (1 - psi) ** t
            if P.mcnemar_exact(w, l) <= alpha:
                total += prob
    assert P.power_exact(n, psi, delta, alpha) == pytest.approx(total)


def test_mde_shrinks_with_n_and_close_to_normal_approx():
    m80, m200 = P.mde_exact(80, 0.15), P.mde_exact(200, 0.15)
    assert m200 < m80
    assert abs(m80 - P.mde_normal(80, 0.15)) < 0.01


def test_rejects_delta_above_psi():
    with pytest.raises(ValueError):
        P.power_exact(80, 0.1, 0.2)
