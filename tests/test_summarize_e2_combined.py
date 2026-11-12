"""E2 多次运行合并汇总(summarize_e2_combined.py):合并配对比较、按清单与层级 × 模式分层、一致性检验、Bonferroni。"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "agent_error"))
import run_e2 as E  # noqa: E402
import summarize_e2_combined as SC  # noqa: E402


def test_fisher_exact_two_sided_known_values():
    assert SC.fisher_exact(3, 1, 1, 3) == pytest.approx(34 / 70)          # 女士品茶:p = 0.4857
    assert SC.fisher_exact(1, 9, 11, 3) == pytest.approx(0.002759, abs=1e-6)
    assert SC.fisher_exact(0, 0, 5, 3) == 1.0                              # 一行全空:无信息
    assert SC.fisher_exact(4, 0, 0, 4) == pytest.approx(2 / 70)


def test_bonferroni():
    assert SC.bonferroni(0.0003, 2) == pytest.approx(0.0006)
    assert SC.bonferroni(0.7, 2) == 1.0
    with pytest.raises(ValueError):
        SC.bonferroni(0.1, 0)


def _pairs(prefix, tier, mode, n):
    return [E.Pair(tier, f"{prefix}_task_{i:03d}", mode) for i in range(n)]


def _parts():
    a = _pairs("C3", "c3", "P3", 4) + _pairs("C4", "c4", "P4", 4)
    b = [E.Pair("c3", f"C3_task_{i:03d}", "P3") for i in range(10, 16)]
    oa = {p.key: o for p, o in zip(a, [(False, True)] * 3 + [(True, False)] + [(False, False)] * 3 + [(True, True)])}
    ob = {p.key: o for p, o in zip(b, [(False, True)] * 5 + [(False, False)])}
    return {"main": (a, oa), "supp": (b, ob)}


def test_combine_totals_parts_and_strata():
    c = SC.combine(_parts())
    t = c["total"]
    assert (t["n"], t["wins"], t["losses"], t["both"]) == (14, 8, 1, 1)
    assert (c["parts"]["main"]["wins"], c["parts"]["main"]["losses"]) == (3, 1)
    assert (c["parts"]["supp"]["wins"], c["parts"]["supp"]["losses"]) == (5, 0)
    assert c["strata"][("c3", "P3")]["n"] == 10 and c["strata"][("c4", "P4")]["n"] == 4
    assert c["strata_parts"][("c3", "P3", "supp")]["wins"] == 5
    # 一致性:两部分不一致配对里胜的比例 3/4 vs 5/5,Fisher 精确
    assert c["homogeneity"] == pytest.approx(SC.fisher_exact(3, 1, 5, 0))


def test_combine_rejects_overlapping_parts():
    parts = _parts()
    a, oa = parts["main"]
    parts["supp"] = (a[:1], {a[0].key: (True, True)})
    with pytest.raises(ValueError):
        SC.combine(parts)


def test_render_states_post_hoc_extension_and_bonferroni():
    c = SC.combine(_parts())
    text = SC.render(c, "test", first="main", decision="扩到 14 对的决定是在看到 main 的结果之后做的。")
    assert "看到 main 的结果之后" in text
    assert "Bonferroni" in text and f"{SC.bonferroni(c['total']['p'], 2):.3g}" in text
    assert f"{SC.bonferroni(c['parts']['main']['p'], 2):.3g}" in text           # 第一次检验也给校正值
    assert "| main |" in text and "| supp |" in text and "| 合计 |" in text
    assert "C3 | P3" in text and "Fisher" in text
