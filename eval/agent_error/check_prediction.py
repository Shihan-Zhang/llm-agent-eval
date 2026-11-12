"""启动前检查:predictions.md 里有没有某个运行的预测行(规矩 2:先写预测再跑)。

预测行的格式是「标签 + 空白 + 内容」,标签在行首,例如「① v2\t基线和回执两组……」。
只有标签、后面为空,不算写好。

用法:python check_prediction.py eval/results/predictions.md "① v2"
退出码 0 = 有;1 = 没有或文件不存在。
"""
from __future__ import annotations

import re
import sys


def has_prediction(text: str, label: str) -> bool:
    pattern = re.compile(rf"^{re.escape(label)}[ \t]+\S", re.MULTILINE)
    return bool(pattern.search(text))


def main(argv: list[str]) -> int:
    path, label = argv
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"读不到 {path}:{exc}", file=sys.stderr)
        return 1
    if has_prediction(text, label):
        print(f"已找到「{label}」的预测")
        return 0
    print(f"{path} 里没有「{label}」的预测行(行首写「{label}」,后面跟内容)。先写预测再启动。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
