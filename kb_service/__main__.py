"""启动知识库检索服务:python -m kb_service(配置从环境变量读,scripts/kb_service.ps1 会先载入 .env)。

只监听 127.0.0.1:索引来自个人笔记(规矩 5),不对外开放;需要开放请自行改代码并先评估风险。
"""
from __future__ import annotations

import functools
import logging
import os

import uvicorn

from kb_service.app import Settings, create_app
from rag.index import ROOT, embed
from rag.retrieve import Retriever

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
LOG_FILE = os.path.join(ROOT, "external", "logs", "kb_service.log")


def main() -> None:
    settings = Settings.from_env(dict(os.environ))          # 没有 Key 时这里就报错退出
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("kb_service")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    retriever = Retriever.load(cpu=True, embed_fn=functools.partial(embed, timeout=settings.embed_timeout_s))
    port = int(os.environ.get("KB_SERVICE_PORT") or DEFAULT_PORT)
    print(f"kb_service: {len(retriever.chunks)} 块,http://{HOST}:{port}(日志 {LOG_FILE})")
    uvicorn.run(create_app(retriever, settings), host=HOST, port=port, access_log=False)


if __name__ == "__main__":
    main()
