from __future__ import annotations

import hashlib

import numpy as np


def stable_hash_features(labels: list[str], dim: int = 64) -> np.ndarray:
    """Deterministic text-feature encoder for perturbation/context identifiers."""
    out = np.zeros((len(labels), dim), dtype=np.float32)
    for i, label in enumerate(labels):
        text = str(label)
        toks = [text]
        toks += text.replace("+", "_").replace(",", "_").replace("-", "_").split("_")
        for tok in toks:
            if not tok:
                continue
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for j in range(0, min(len(digest), dim), 2):
                idx = digest[j] % dim
                sign = 1.0 if digest[j + 1] % 2 else -1.0
                out[i, idx] += sign
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


PATHWAY_HINTS = {
    "cell_cycle": ["CDK", "CCN", "MKI", "E2F", "MYC", "AURK", "PLK"],
    "immune_ifn": ["IFN", "STAT", "IRF", "JAK", "ISG", "CXCL", "IL"],
    "stress_apoptosis": ["TP53", "BAX", "CASP", "BCL", "ATF", "JUN", "FOS", "DDIT"],
    "chromatin": ["HDAC", "KDM", "DNMT", "EZH", "EP300", "SMAR", "BRD"],
    "metabolism": ["MTOR", "AMPK", "PPAR", "SLC", "LDH", "G6P", "ACLY"],
    "tf_program": ["SOX", "GATA", "FOXP", "RUNX", "NFKB", "CEBP", "KLF", "POU"],
}


def pathway_prior_features(perturbations: list[str], contexts: list[str], dim: int = 64) -> np.ndarray:
    """Small deterministic pathway/graph prior features.

    This is intentionally lightweight: it maps perturbation/context tokens into
    coarse pathway buckets plus hashed interaction terms.  It makes V2 different
    from V1 in kind, not merely in ridge alpha.
    """
    n = len(perturbations)
    buckets = list(PATHWAY_HINTS)
    out = np.zeros((n, len(buckets) + dim), dtype=np.float32)
    for i, (pert, ctx) in enumerate(zip(perturbations, contexts)):
        text = f"{pert} {ctx}".upper()
        for j, name in enumerate(buckets):
            if any(tok in text for tok in PATHWAY_HINTS[name]):
                out[i, j] = 1.0
        inter = stable_hash_features([f"{pert}::{ctx}"], dim=dim)[0]
        out[i, len(buckets) :] = inter
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


def nearest_prior_smoothing(
    query_prior: np.ndarray,
    train_prior: np.ndarray,
    train_effects: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Graph-style smoothing over nearest pathway-prior neighbors."""
    if len(train_prior) == 0:
        return np.zeros((len(query_prior), train_effects.shape[1]), dtype=np.float32)
    q = query_prior / np.maximum(np.linalg.norm(query_prior, axis=1, keepdims=True), 1e-8)
    t = train_prior / np.maximum(np.linalg.norm(train_prior, axis=1, keepdims=True), 1e-8)
    sim = q @ t.T
    kk = min(k, sim.shape[1])
    idx = np.argsort(-sim, axis=1)[:, :kk]
    rows = []
    for r, ids in enumerate(idx):
        w = np.maximum(sim[r, ids], 0)
        if float(w.sum()) <= 1e-8:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        rows.append((train_effects[ids] * w[:, None]).sum(axis=0))
    return np.asarray(rows, dtype=np.float32)


def standardize_train_apply(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std
