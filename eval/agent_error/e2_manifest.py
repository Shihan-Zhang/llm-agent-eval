"""
冻结 E2 的回合清单(在看到任何 E2 结果之前生成,之后不改;哈希记在 eval/agent_error/README.md)。

主实验按论文(arXiv 2608.19303 §4)抽样:C3/P3、C3/P4、C4/P3、C4/P4 各 20 个,共 80 个互不相同的任务编号,
每个任务只带一种故障模式。补充实验:C1 全部 100 题 × P3/P4(不抽样)。

用法(仓库根目录):python eval/agent_error/e2_manifest.py
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


def main() -> None:
    if os.path.exists(OUT):
        sys.exit(f"{OUT} 已存在;清单冻结后不再重新生成")
    m = build()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
    print(f"main {len(m['main'])} 回合,c1_supplement {len(m['c1_supplement'])} 回合,sha256 {digest(m)}")


if __name__ == "__main__":
    main()
