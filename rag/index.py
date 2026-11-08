"""
建索引:chunks.jsonl → bge-m3 向量 rag/store/emb.npy + manifest.json

- 向量:Ollama /api/embed,按批发送 embed_text;结果逐行归一化(之后点积 = 余弦相似度),float32。
  N 块 × 1024 维 × 4 字节;1 万块约 40 MB,numpy 暴力检索足够,不需要向量数据库。
- BM25 不落盘:从 chunks.jsonl 现建只要几百毫秒,省掉序列化格式和版本问题。
- manifest 记录 embedding 模型、Ollama 版本、chunks.jsonl 的哈希、维度、切分参数,
  评测结果要能追溯到具体是哪个版本的索引;检索时会核对哈希,块和向量对不上就报错。

建索引是离线的一次性工作:默认用 GPU,跑完卸载 bge-m3,给生成模型腾显存。
运行:python -m rag.index [--cpu]
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "rag", "store")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA


def _post(path, body, timeout=600):
    req = urllib.request.Request(f"{OLLAMA}{path}", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed(texts, model="bge-m3", cpu=False, keep_alive="5m"):
    """返回 [len(texts), dim] 的 float32 矩阵,逐行归一化。"""
    body = {"model": model, "input": texts, "keep_alive": keep_alive}
    if cpu:
        body["options"] = {"num_gpu": 0}           # 0 层放 GPU,即全部在 CPU 上算
    vecs = np.asarray(_post("/api/embed", body)["embeddings"], dtype=np.float32)
    if vecs.ndim != 2 or len(vecs) != len(texts):
        raise ValueError(f"embedding 形状异常:{vecs.shape},期望 ({len(texts)}, dim)")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    if not np.all(np.isfinite(vecs)) or np.any(norms == 0):
        raise ValueError("embedding 含 NaN/Inf 或零向量")
    return vecs / norms


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=os.path.join(STORE, "chunks.jsonl"))
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cpu", action="store_true", help="在 CPU 上算向量(显存不够时)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    with open(args.chunks, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    texts = [c["embed_text"] for c in chunks]
    t0, parts = time.time(), []
    for i in range(0, len(texts), args.batch):
        parts.append(embed(texts[i:i + args.batch], args.model, args.cpu))
        print(f"  {min(i + args.batch, len(texts))}/{len(texts)}", end="\r")
    emb = np.vstack(parts)
    secs = time.time() - t0
    _post("/api/embed", {"model": args.model, "input": ["x"], "keep_alive": 0})   # 卸载,腾显存

    out_dir = os.path.dirname(args.chunks)
    np.save(os.path.join(out_dir, "emb.npy"), emb)
    with urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=10) as resp:
        version = json.loads(resp.read().decode("utf-8")).get("version")
    manifest = {"embedding_model": args.model, "ollama_version": version, "device": "cpu" if args.cpu else "gpu",
                "n_chunks": len(chunks), "dim": int(emb.shape[1]), "embed_field": "embed_text",
                "chunks_sha256": file_sha256(args.chunks), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "embed_seconds": round(secs, 1),
                "max_tokens_est": max(c["tokens_est"] for c in chunks)}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n{len(chunks)} 块 × {emb.shape[1]} 维,{secs:.1f} s({len(chunks) / max(secs, 1e-9):.1f} 块/秒)"
          f",{emb.nbytes / 1e6:.1f} MB → {out_dir}")


if __name__ == "__main__":
    main()
