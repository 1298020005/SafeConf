from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF


class ProgramBank:
    """Multi-view program bank for compact perturbation-effect transport.

    Earlier V2 used only an SVD/PCA-like bank.  This version explicitly supports
    a combined PCA + NMF + high-variance-gene (HVG) basis so that effect-gene
    support metrics are not forced through a single low-rank subspace.
    """

    def __init__(self, n_programs: int = 64, seed: int = 0, mode: str = "pca"):
        self.n_programs = int(n_programs)
        self.seed = int(seed)
        self.mode = str(mode)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.component_blocks_: dict[str, int] = {}

    def fit(self, effects: np.ndarray) -> "ProgramBank":
        x = np.asarray(effects, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("effects must be 2D")
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        blocks: list[np.ndarray] = []
        self.component_blocks_ = {}

        def add_block(name: str, comp: np.ndarray) -> None:
            if comp.size == 0:
                return
            # Normalize rows for comparable coefficient scales.
            norm = np.linalg.norm(comp, axis=1, keepdims=True)
            norm[norm < 1e-8] = 1.0
            comp = comp / norm
            blocks.append(comp)
            self.component_blocks_[name] = comp.shape[0]

        if self.mode in {"pca", "combo", "pca_nmf_hvg"}:
            _, _, vt = np.linalg.svd(xc, full_matrices=False)
            n = min(max(2, self.n_programs // (3 if self.mode != "pca" else 1)), vt.shape[0])
            add_block("pca", vt[:n])

        if self.mode in {"nmf", "combo", "pca_nmf_hvg"}:
            # NMF is fit on effect magnitudes; signs remain represented by PCA/HVG
            # and by signed inverse coefficients at prediction time.
            mag = np.abs(xc)
            n = min(max(2, self.n_programs // 3), mag.shape[0], mag.shape[1])
            if n >= 2 and np.any(mag > 0):
                nmf = NMF(n_components=n, init="nndsvda", random_state=self.seed, max_iter=1400, tol=1e-4)
                nmf.fit(mag + 1e-6)
                add_block("nmf_magnitude", nmf.components_)

        if self.mode in {"hvg", "combo", "pca_nmf_hvg"}:
            n = min(max(2, self.n_programs // 3), x.shape[1])
            idx = np.argsort(-np.var(xc, axis=0))[:n]
            eye = np.zeros((len(idx), x.shape[1]), dtype=np.float64)
            eye[np.arange(len(idx)), idx] = 1.0
            add_block("hvg_identity", eye)

        if not blocks:
            _, _, vt = np.linalg.svd(xc, full_matrices=False)
            add_block("pca_fallback", vt[: min(self.n_programs, vt.shape[0])])

        comp = np.vstack(blocks)
        if comp.shape[0] > self.n_programs:
            comp = comp[: self.n_programs]
        self.components_ = comp.astype(np.float32)
        return self

    def transform(self, effects: np.ndarray) -> np.ndarray:
        self._check()
        x = np.asarray(effects, dtype=np.float64)
        return ((x - self.mean_) @ self.components_.T).astype(np.float32)

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        self._check()
        z = np.asarray(coeffs, dtype=np.float64)
        return (z @ self.components_ + self.mean_).astype(np.float32)

    def _check(self) -> None:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("ProgramBank is not fitted")
