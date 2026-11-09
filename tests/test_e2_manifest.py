"""E2 回合清单:② 的补充清单只用主清单没用过的任务、每题一种模式、各层平分,且必须基于同一份主清单。"""
import json
import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "agent_error"))
import e2_manifest as M  # noqa: E402
import run_e2 as E  # noqa: E402


def fake_toolmaze(tmp_path, per_tier=12):
    for tier in M.MAIN_TIERS:
        d = tmp_path / "data" / "perturbed_tasks" / tier
        d.mkdir(parents=True)
        for i in range(per_tier):
            for mode in ("P0",) + M.MODES:
                (d / f"{tier.upper()}_task_{i:03d}_{mode}.json").write_text("{}", encoding="utf-8")
    return str(tmp_path)


def test_ext_uses_only_unused_tasks_balanced_one_mode_each(tmp_path):
    tm = fake_toolmaze(tmp_path)
    base = M.build(tm, seed=1, n=2)                       # 每层用掉 4 题,剩 8 题
    ext = M.build_ext(base, tm, seed=2)
    used = {e["task_id"] for e in base["main"]}
    ids = [e["task_id"] for e in ext[M.EXT_KEY]]
    assert len(ids) == len(set(ids)) == 16 and not used & set(ids)
    assert Counter((e["tier"], e["mode"]) for e in ext[M.EXT_KEY]) == {(t, m): 4 for t in M.MAIN_TIERS for m in M.MODES}
    assert ext["base_sha256"] == M.digest(base)
    assert M.build_ext(base, tm, seed=2) == ext           # 固定种子可复现


def test_load_merges_ext_and_keeps_base_hash(tmp_path):
    tm = fake_toolmaze(tmp_path)
    base = M.build(tm, seed=1, n=2)
    base_p, ext_p = tmp_path / "m.json", tmp_path / "ext.json"
    base_p.write_text(json.dumps(base), encoding="utf-8")
    ext = M.build_ext(base, tm, seed=2)
    ext_p.write_text(json.dumps(ext), encoding="utf-8")
    m = M.load(str(base_p), str(ext_p))
    assert M.hashes(m) == {"manifest_sha256": M.digest(base), "ext_sha256": M.digest(ext)}
    assert len(E.pairs_for("main_ext", m)) == len(base["main"]) + 16
    assert len(E.pairs_for("main", m)) == len(base["main"])
    assert M.hashes(M.load(str(base_p), str(tmp_path / "missing.json"))) == {"manifest_sha256": M.digest(base)}


def test_load_rejects_ext_built_from_another_base(tmp_path):
    tm = fake_toolmaze(tmp_path)
    base = M.build(tm, seed=1, n=2)
    other = M.build(tm, seed=3, n=2)
    base_p, ext_p = tmp_path / "m.json", tmp_path / "ext.json"
    base_p.write_text(json.dumps(base), encoding="utf-8")
    ext_p.write_text(json.dumps(M.build_ext(other, tm, seed=2)), encoding="utf-8")
    with pytest.raises(ValueError):
        M.load(str(base_p), str(ext_p))


def test_frozen_files_in_repo():
    m = M.load()
    assert M.hashes(m) == {"manifest_sha256": "8d43264736a91674342f3a828a9fe9d511df8b2314b7e48af6d31be38916936d",
                           "ext_sha256": "eb47ca3ac1b8d419d901e06989d2e8ad671a878ba68cd3aae184d6bb429f4632"}
    assert len(E.pairs_for("main_ext", m)) == 200
