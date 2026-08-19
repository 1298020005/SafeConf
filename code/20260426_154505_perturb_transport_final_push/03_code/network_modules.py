from __future__ import annotations

import numpy as np
import pandas as pd


class NetworkModuleBank:
    """hdWGCNA-inspired co-expression module bank.

    This is a lightweight Python implementation for the perturbation-transport
    runners.  It does not try to replace the hdWGCNA R package; it borrows the
    useful idea for this project: compress gene-level effects into modules that
    are defined by soft-thresholded co-expression connectivity and hub genes.
    """

    def __init__(
        self,
        n_modules: int = 96,
        seed: int = 0,
        soft_power: int = 6,
        max_network_genes: int = 1600,
        min_module_size: int = 8,
        hvg_fraction: float = 0.25,
    ):
        self.n_modules = int(n_modules)
        self.seed = int(seed)
        self.soft_power = int(soft_power)
        self.max_network_genes = int(max_network_genes)
        self.min_module_size = int(min_module_size)
        self.hvg_fraction = float(hvg_fraction)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.module_table_: pd.DataFrame | None = None
        self.component_blocks_: dict[str, int] = {}

    def fit(self, effects: np.ndarray) -> "NetworkModuleBank":
        x = np.asarray(effects, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("effects must be 2D")
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        n_tasks, n_genes = xc.shape
        if n_tasks < 4 or n_genes < 16:
            return self._fit_hvg_fallback(xc)

        var = np.var(xc, axis=0)
        n_candidates = min(n_genes, max(32, self.max_network_genes))
        candidate_idx = np.argsort(-var)[:n_candidates]
        z = xc[:, candidate_idx]
        z = z - z.mean(axis=0, keepdims=True)
        std = z.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        z = z / std

        corr = (z.T @ z) / max(1, n_tasks - 1)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        adjacency = np.abs(corr) ** self.soft_power
        np.fill_diagonal(adjacency, 0.0)

        components: list[np.ndarray] = []
        module_rows: list[dict] = []
        network_budget = max(2, int(round(self.n_modules * (1.0 - self.hvg_fraction))))
        seed_positions = self._choose_seed_positions(var[candidate_idx], corr, network_budget)
        if len(seed_positions) == 0:
            return self._fit_hvg_fallback(xc)

        seed_adj = adjacency[:, seed_positions]
        assigned = np.argmax(seed_adj, axis=1)
        strength = seed_adj[np.arange(seed_adj.shape[0]), assigned]
        used = np.zeros(n_candidates, dtype=bool)

        for module_id, seed_pos in enumerate(seed_positions):
            members = np.flatnonzero((assigned == module_id) & (strength > 0))
            if len(members) < self.min_module_size:
                top = np.argsort(-adjacency[:, seed_pos])[: self.min_module_size]
                members = np.unique(np.concatenate([members, top, np.array([seed_pos])]))
            members = members[~used[members]]
            if len(members) < max(3, self.min_module_size // 2):
                continue
            used[members] = True
            comp, rows = self._module_component(
                module_id=module_id,
                candidate_idx=candidate_idx,
                members=members,
                z=z,
                corr=corr,
                adjacency=adjacency,
                n_genes=n_genes,
            )
            components.append(comp)
            module_rows.extend(rows)
            if len(components) >= network_budget:
                break

        hvg_budget = self.n_modules - len(components)
        if hvg_budget > 0:
            for gene_idx in np.argsort(-var)[:hvg_budget]:
                comp = np.zeros(n_genes, dtype=np.float64)
                comp[int(gene_idx)] = 1.0
                components.append(comp)
                module_rows.append(
                    {
                        "module": f"HVG_{len(components):03d}",
                        "gene_index": int(gene_idx),
                        "hub_weight": 1.0,
                        "kind": "hvg_identity",
                    }
                )

        if not components:
            return self._fit_hvg_fallback(xc)

        comp = np.vstack(components)
        norm = np.linalg.norm(comp, axis=1, keepdims=True)
        norm[norm < 1e-8] = 1.0
        self.components_ = (comp / norm).astype(np.float32)
        self.component_blocks_ = {"network_modules": min(len(components), network_budget)}
        if len(components) > network_budget:
            self.component_blocks_["hvg_identity"] = len(components) - network_budget
        self.module_table_ = pd.DataFrame(module_rows)
        return self

    def _choose_seed_positions(self, variance: np.ndarray, corr: np.ndarray, budget: int) -> np.ndarray:
        order = np.argsort(-variance)
        seeds: list[int] = []
        for pos in order:
            if len(seeds) >= budget:
                break
            if not seeds:
                seeds.append(int(pos))
                continue
            max_corr = float(np.max(np.abs(corr[int(pos), seeds])))
            if max_corr < 0.72:
                seeds.append(int(pos))
        if len(seeds) < max(2, budget // 2):
            for pos in order:
                if len(seeds) >= budget:
                    break
                if int(pos) not in seeds:
                    seeds.append(int(pos))
        return np.asarray(seeds[:budget], dtype=int)

    def _module_component(
        self,
        module_id: int,
        candidate_idx: np.ndarray,
        members: np.ndarray,
        z: np.ndarray,
        corr: np.ndarray,
        adjacency: np.ndarray,
        n_genes: int,
    ) -> tuple[np.ndarray, list[dict]]:
        module_name = f"NETM_{module_id + 1:03d}"
        eig = z[:, members].mean(axis=1)
        eig_std = float(np.std(eig))
        if eig_std < 1e-8:
            eig = z[:, members[0]]
            eig_std = float(np.std(eig)) or 1.0
        signed = (z[:, members].T @ eig) / max(1, z.shape[0] - 1)
        signed = signed / max(eig_std, 1e-8)
        within = adjacency[np.ix_(members, members)]
        hub = within.mean(axis=1) if within.size else np.ones(len(members))
        hub = np.nan_to_num(hub, nan=0.0)
        if float(np.max(hub)) > 0:
            hub = hub / float(np.max(hub))
        weights = np.sign(signed) * (0.35 + 0.65 * hub)
        comp = np.zeros(n_genes, dtype=np.float64)
        full_idx = candidate_idx[members]
        comp[full_idx] = weights
        rows = []
        for gene_idx, weight, hub_score in zip(full_idx, weights, hub):
            rows.append(
                {
                    "module": module_name,
                    "gene_index": int(gene_idx),
                    "hub_weight": float(abs(weight)),
                    "signed_weight": float(weight),
                    "hub_score": float(hub_score),
                    "kind": "network_module",
                }
            )
        return comp, rows

    def _fit_hvg_fallback(self, centered: np.ndarray) -> "NetworkModuleBank":
        n_genes = centered.shape[1]
        var = np.var(centered, axis=0)
        n = min(self.n_modules, n_genes)
        idx = np.argsort(-var)[:n]
        comp = np.zeros((n, n_genes), dtype=np.float64)
        comp[np.arange(n), idx] = 1.0
        self.components_ = comp.astype(np.float32)
        self.component_blocks_ = {"hvg_fallback": n}
        self.module_table_ = pd.DataFrame(
            [
                {"module": f"HVG_{i + 1:03d}", "gene_index": int(g), "hub_weight": 1.0, "kind": "hvg_fallback"}
                for i, g in enumerate(idx)
            ]
        )
        return self

    def transform(self, effects: np.ndarray) -> np.ndarray:
        self._check()
        x = np.asarray(effects, dtype=np.float64)
        return ((x - self.mean_) @ self.components_.T).astype(np.float32)

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        self._check()
        z = np.asarray(coeffs, dtype=np.float64)
        return (z @ self.components_ + self.mean_).astype(np.float32)

    def module_table(self) -> pd.DataFrame:
        self._check()
        return self.module_table_.copy() if self.module_table_ is not None else pd.DataFrame()

    def _check(self) -> None:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("NetworkModuleBank is not fitted")
