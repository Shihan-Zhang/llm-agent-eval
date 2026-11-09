"""
冻结 E2 的回合清单(在看到任何 E2 结果之前生成,之后不改;哈希记在 eval/agent_error/README.md)。

主实验按论文(arXiv 2608.19303 §4)抽样:C3/P3、C3/P4、C4/P3、C4/P4 各 20 个,共 80 个互不相同的任务编号,
每个任务只带一种故障模式。补充实验:C1 全部 100 题 × P3/P4(不抽样)。
② 的补充清单(2026-11-09,② 运行前):C3/C4 里主清单没用过的 120 题,同样每题一种模式,存 configs/e2_manifest_ext.json。

用法(仓库根目录):python eval/agent_error/e2_manifest.py [--ext]
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TM = os.environ.get("TOOLMAZE_DIR", os.path.join(ROOT, "external", "agent_error", "toolmaze"))
OUT = os.path.join(HERE, "configs", "e2_manifest.json")
SEED = 20261108
PER_STRATUM = 20
MAIN_TIERS = ("c3", "c4")
MODES = ("P3", "P4")
EXT_OUT = os.path.join(HERE, "configs", "e2_manifest_ext.json")   # ② 的补充清单,另存一份,主清单不动
EXT_SEED = 20261109
EXT_KEY = "stripped_ext"
EXT_HASH_KEY = "stripped_ext_sha256"   # load() 并入时记下补充清单文件的哈希


def tasks_with_modes(tm_dir: str, tier: str, modes: tuple[str, ...] = MODES) -> list[str]:
    """该层级里 P0 与给定模式的题目文件都存在的任务编号(排序)。"""
    ids = []
    for f in sorted(glob.glob(os.path.join(tm_dir, "data", "perturbed_tasks", tier, "*_P0.json"))):
        task_id = os.path.basename(f)[: -len("_P0.json")]
        if all(os.path.exists(os.path.join(os.path.dirname(f), f"{task_id}_{m}.json")) for m in modes):
            ids.append(task_id)
    return ids


def build(tm_dir: str = TM, seed: int = SEED, n: int = PER_STRATUM) -> dict:
    rng = random.Random(seed)
    main = []
    for tier in MAIN_TIERS:
        ids = tasks_with_modes(tm_dir, tier)
        if len(ids) < n * len(MODES):
            raise ValueError(f"{tier} 可用任务 {len(ids)} 个,不够 {n * len(MODES)} 个")
        picked = rng.sample(ids, n * len(MODES))      # 同一层级内任务互不重复
        for i, mode in enumerate(MODES):
            main += [{"tier": tier, "task_id": t, "mode": mode} for t in sorted(picked[i * n:(i + 1) * n])]
    c1 = [{"tier": "c1", "task_id": t, "mode": m} for t in tasks_with_modes(tm_dir, "c1") for m in MODES]
    return {"seed": seed, "per_stratum": n, "main": main, "c1_supplement": c1}


def digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_ext(base: dict, tm_dir: str = TM, seed: int = EXT_SEED) -> dict:
    """② 的补充清单:C3/C4 里主清单没用过的任务全部取出,每层打乱后一半 P3、一半 P4(每题仍只带一种模式)。"""
    rng = random.Random(seed)
    used = {e["task_id"] for e in base["main"]}
    ext = []
    for tier in MAIN_TIERS:
        unused = [t for t in tasks_with_modes(tm_dir, tier) if t not in used]
        if not unused or len(unused) % len(MODES):
            raise ValueError(f"{tier} 剩余任务 {len(unused)} 个,不能平分到 {len(MODES)} 种模式")
        picked = rng.sample(unused, len(unused))
        n = len(unused) // len(MODES)
        for i, mode in enumerate(MODES):
            ext += [{"tier": tier, "task_id": t, "mode": mode} for t in sorted(picked[i * n:(i + 1) * n])]
    return {"seed": seed, "base_sha256": digest(base), EXT_KEY: ext}


def load(path: str = OUT, ext_path: str = EXT_OUT) -> dict:
    """冻结清单;补充清单存在时并入 EXT_KEY(先核对它是基于这份主清单生成的)。"""
    with open(path, encoding="utf-8") as fh:
        base = json.load(fh)
    if not os.path.exists(ext_path):
        return base
    with open(ext_path, encoding="utf-8") as fh:
        ext = json.load(fh)
    if ext["base_sha256"] != digest(base):
        raise ValueError(f"{ext_path} 的 base_sha256 与主清单不符")
    return {**base, EXT_KEY: ext[EXT_KEY], EXT_HASH_KEY: digest(ext)}


def hashes(manifest: dict) -> dict:
    """运行记录里写的清单哈希:主清单一份;并入了补充清单时,补充部分另记一份。"""
    base = {k: v for k, v in manifest.items() if k not in (EXT_KEY, EXT_HASH_KEY)}
    out = {"manifest_sha256": digest(base)}
    if EXT_HASH_KEY in manifest:
        out["ext_sha256"] = manifest[EXT_HASH_KEY]
    return out


def write_frozen(path: str, m: dict) -> None:
    if os.path.exists(path):
        sys.exit(f"{path} 已存在;清单冻结后不再重新生成")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)


def main() -> None:
    if "--ext" in sys.argv[1:]:
        m = build_ext(load(OUT, ext_path=""))
        write_frozen(EXT_OUT, m)
        print(f"{EXT_KEY} {len(m[EXT_KEY])} 回合,sha256 {digest(m)}")
        return
    m = build()
    write_frozen(OUT, m)
    print(f"main {len(m['main'])} 回合,c1_supplement {len(m['c1_supplement'])} 回合,sha256 {digest(m)}")


if __name__ == "__main__":
    main()
