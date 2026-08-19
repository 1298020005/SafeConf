#!/usr/bin/env python3
"""Prepare local, provenance-tracked scGPT embeddings for PRESCRIBE."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"
VOCAB = CHECKPOINT_DIR / "vocab.json"
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
ASSET_DIR = PRESCRIBE / "scLLM_weights/scGPT"
OUT = ROOT / "docs/实验结果/E92_prescribe_assets_20260712"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    vocab = json.loads(VOCAB.read_text())
    state = torch.load(CHECKPOINT, map_location="cpu")
    weights = state["encoder.embedding.weight"].float()
    gamma = state["encoder.enc_norm.weight"].float()
    beta = state["encoder.enc_norm.bias"].float()
    if weights.shape[0] != len(vocab):
        raise RuntimeError(f"Checkpoint/vocab mismatch: {weights.shape[0]} vs {len(vocab)}")
    normalized = torch.nn.functional.layer_norm(
        weights, (weights.shape[1],), weight=gamma, bias=beta, eps=1e-5
    ).cpu().numpy().astype(np.float32)
    embedding = {}
    for token, index in vocab.items():
        index = int(index)
        if index < 0 or index >= len(normalized):
            raise RuntimeError(f"Invalid vocabulary index for {token}: {index}")
        embedding[str(token)] = normalized[index]
    if len(embedding) != len(vocab):
        raise RuntimeError("Duplicate vocabulary tokens")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    embedding_path = ASSET_DIR / "embedding.pkl"
    gene_embedding_path = ASSET_DIR / "gene_emb.pkl"
    with embedding_path.open("wb") as handle:
        pickle.dump(embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if gene_embedding_path.exists() or gene_embedding_path.is_symlink():
        gene_embedding_path.unlink()
    os.link(embedding_path, gene_embedding_path)

    # Reload from disk before declaring the asset complete.
    with embedding_path.open("rb") as handle:
        check = pickle.load(handle)
    if len(check) != len(vocab) or np.asarray(check["<pad>"]).shape != (weights.shape[1],):
        raise RuntimeError("Serialized embedding verification failed")
    status = {
        "experiment": "E92_prescribe_assets",
        "phase": "scgpt_embeddings_ready",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prescribe_commit": "6f7264a205aaff654a9594863c5c10b656f88ebe",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "vocab": str(VOCAB),
        "vocab_sha256": sha256(VOCAB),
        "n_tokens": len(embedding),
        "embedding_dim": int(weights.shape[1]),
        "extraction": "encoder.embedding.weight followed by frozen encoder.enc_norm",
        "embedding_path": str(embedding_path),
        "gene_embedding_path": str(gene_embedding_path),
        "same_inode": embedding_path.stat().st_ino == gene_embedding_path.stat().st_ino,
        "embedding_sha256": sha256(embedding_path),
        "downloaded_google_drive_embedding_used": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README_先看这个.md").write_text(
        "# E92｜PRESCRIBE 本地 scGPT 资产\n\n"
        "基因嵌入直接从本机 whole-human scGPT 冻结权重与对应词表提取，并经过原模型的 `encoder.enc_norm`。"
        "两个 PRESCRIBE 所需文件为同一 inode，避免保存两份约百 MB 的重复 pickle。完整哈希见 `RUN_STATUS.json`。\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
