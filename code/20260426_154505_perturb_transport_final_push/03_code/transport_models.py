from __future__ import annotations

import os

import numpy as np

from encoders import nearest_prior_smoothing, pathway_prior_features, stable_hash_features, standardize_train_apply


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        xs, _, mean, std = standardize_train_apply(np.asarray(x, dtype=np.float64), np.asarray(x, dtype=np.float64))
        yy = np.asarray(y, dtype=np.float64)
        self.y_mean_ = yy.mean(axis=0)
        yc = yy - self.y_mean_
        xtx = xs.T @ xs
        reg = self.alpha * np.eye(xtx.shape[0])
        self.coef_ = np.linalg.solve(xtx + reg, xs.T @ yc)
        self.x_mean_ = mean
        self.x_std_ = std
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.x_mean_ is None or self.x_std_ is None or self.y_mean_ is None:
            raise RuntimeError("RidgeRegressor is not fitted")
        xs = (np.asarray(x, dtype=np.float64) - self.x_mean_) / self.x_std_
        return (xs @ self.coef_ + self.y_mean_).astype(np.float32)


class V0StrongBaseline:
    """Strong effect baseline: same-perturbation mean plus target-context residual."""

    name = "V0"

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "V0StrongBaseline":
        self.global_mean = np.mean([t["effect"] for t, keep in zip(tasks, train_mask) if keep], axis=0)
        self.by_pert = {}
        self.by_context = {}
        for t, keep in zip(tasks, train_mask):
            if not keep:
                continue
            self.by_pert.setdefault(t["perturbation"], []).append(t["effect"])
            self.by_context.setdefault(t["context"], []).append(t["effect"])
        self.by_pert = {k: np.mean(v, axis=0) for k, v in self.by_pert.items()}
        self.by_context = {k: np.mean(v, axis=0) for k, v in self.by_context.items()}
        return self

    def predict_one(self, task: dict) -> np.ndarray:
        pred = self.by_pert.get(task["perturbation"], self.global_mean).copy()
        if task["context"] in self.by_context:
            pred = 0.85 * pred + 0.15 * self.by_context[task["context"]]
        return pred.astype(np.float32)

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        return np.stack([self.predict_one(tasks[int(i)]) for i in indices], axis=0)


class V1ProgramTransport:
    """Program-level transport from source-context effects to target-context effects."""

    name = "V1"

    def __init__(self, program_bank, alpha: float = 3.0, hash_dim: int = 64):
        self.bank = program_bank
        self.alpha = alpha
        self.hash_dim = hash_dim
        self.reg = RidgeRegressor(alpha=alpha)

    def _source_effect(self, task: dict, tasks: list[dict], train_mask: np.ndarray) -> np.ndarray:
        vals = [
            t["effect"]
            for t, keep in zip(tasks, train_mask)
            if keep and t["perturbation"] == task["perturbation"] and t["context"] != task["context"]
        ]
        if vals:
            return np.mean(vals, axis=0)
        vals = [t["effect"] for t, keep in zip(tasks, train_mask) if keep and t["perturbation"] == task["perturbation"]]
        if vals:
            return np.mean(vals, axis=0)
        return self.global_effect

    def _features(self, selected: list[dict], tasks: list[dict], train_mask: np.ndarray) -> np.ndarray:
        src = np.stack([self._source_effect(t, tasks, train_mask) for t in selected], axis=0)
        src_z = self.bank.transform(src)
        ctrl_z = self.bank.transform(np.stack([t["control_mean"] for t in selected], axis=0))
        pert_h = stable_hash_features([t["perturbation"] for t in selected], self.hash_dim)
        ctx_h = stable_hash_features([t["context"] for t in selected], self.hash_dim)
        return np.concatenate([src_z, ctrl_z, pert_h, ctx_h], axis=1)

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "V1ProgramTransport":
        train = [t for t, keep in zip(tasks, train_mask) if keep]
        self.baseline = V0StrongBaseline().fit(tasks, train_mask)
        self.train_mask_for_predict = train_mask.copy()
        self.global_effect = np.mean([t["effect"] for t in train], axis=0)
        self.bank.fit(np.stack([t["effect"] for t in train], axis=0))
        x = self._features(train, tasks, train_mask)
        y = self.bank.transform(np.stack([t["effect"] for t in train], axis=0))
        self.reg.fit(x, y)
        return self

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        selected = [tasks[int(i)] for i in indices]
        train_mask = getattr(self, "train_mask_for_predict", None)
        if train_mask is None:
            train_mask = np.ones(len(tasks), dtype=bool)
        x = self._features(selected, tasks, train_mask)
        z = self.reg.predict(x)
        transported = self.bank.inverse_transform(z)
        baseline = self.baseline.predict(tasks, indices)
        return (0.9 * baseline + 0.1 * transported).astype(np.float32)


class V2GraphPriorTransport(V1ProgramTransport):
    name = "V2"

    def __init__(self, program_bank, alpha: float = 5.0, hash_dim: int = 96, blend: float | None = None):
        super().__init__(program_bank, alpha=alpha, hash_dim=hash_dim)
        self.blend = float(os.environ.get("PAIRDELTA_V2_BLEND", blend if blend is not None else 0.18))

    def _prior(self, selected: list[dict]) -> np.ndarray:
        return pathway_prior_features(
            [t["perturbation"] for t in selected],
            [t["context"] for t in selected],
            dim=self.hash_dim,
        )

    def _features(self, selected: list[dict], tasks: list[dict], train_mask: np.ndarray) -> np.ndarray:
        src = np.stack([self._source_effect(t, tasks, train_mask) for t in selected], axis=0)
        src_z = self.bank.transform(src)
        ctrl_z = self.bank.transform(np.stack([t["control_mean"] for t in selected], axis=0))
        prior = self._prior(selected)
        smooth = nearest_prior_smoothing(
            prior,
            self.train_prior_,
            self.train_effects_,
            k=min(5, len(self.train_effects_)),
        )
        smooth_z = self.bank.transform(smooth)
        return np.concatenate([src_z, ctrl_z, smooth_z, prior, src_z * smooth_z], axis=1)

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "V2GraphPriorTransport":
        train = [t for t, keep in zip(tasks, train_mask) if keep]
        self.baseline = V0StrongBaseline().fit(tasks, train_mask)
        self.train_mask_for_predict = train_mask.copy()
        self.global_effect = np.mean([t["effect"] for t in train], axis=0)
        self.bank.fit(np.stack([t["effect"] for t in train], axis=0))
        self.train_prior_ = self._prior(train)
        self.train_effects_ = np.stack([t["effect"] for t in train], axis=0)
        x = self._features(train, tasks, train_mask)
        y = self.bank.transform(self.train_effects_)
        self.reg.fit(x, y)
        return self

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        selected = [tasks[int(i)] for i in indices]
        train_mask = getattr(self, "train_mask_for_predict", np.ones(len(tasks), dtype=bool))
        x = self._features(selected, tasks, train_mask)
        z = self.reg.predict(x)
        transported = self.bank.inverse_transform(z)
        baseline = self.baseline.predict(tasks, indices)
        # V2 should actually use the graph/pathway transport; previous V2 was
        # effectively a 90% baseline blend.  This stronger blend is still
        # conservative but lets effect-gene support move.
        b = min(max(self.blend, 0.0), 1.0)
        return ((1.0 - b) * baseline + b * transported).astype(np.float32)


class ContextSimilarityBaseline:
    """Reviewer-facing comparator: transport only when source contexts look similar."""

    name = "ContextSimBaseline"

    def __init__(self, temperature: float = 5.0, unsafe_threshold: float = 0.35):
        self.temperature = float(temperature)
        self.unsafe_threshold = float(unsafe_threshold)

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "ContextSimilarityBaseline":
        self.train_mask_for_predict = train_mask.copy()
        train = [t for t, keep in zip(tasks, train_mask) if keep]
        self.train_controls_ = np.stack([t["control_mean"] for t in train], axis=0)
        self.train_effects_ = np.stack([t["effect"] for t in train], axis=0)
        self.train_pert_ = [str(t["perturbation"]) for t in train]
        self.train_ctx_ = [str(t["context"]) for t in train]
        self.global_mean_ = np.mean(self.train_effects_, axis=0)
        return self

    def _predict_one(self, task: dict) -> tuple[np.ndarray, float, int]:
        pert = str(task["perturbation"])
        ctx = str(task["context"])
        idxs = [
            j
            for j, (p, c) in enumerate(zip(self.train_pert_, self.train_ctx_))
            if p == pert and c != ctx
        ]
        if not idxs:
            idxs = [j for j, p in enumerate(self.train_pert_) if p == pert]
        if not idxs:
            return self.global_mean_.astype(np.float32), 0.0, 1

        query = np.asarray(task["control_mean"], dtype=np.float64)
        sims = []
        for j in idxs:
            ref = self.train_controls_[j]
            sims.append(
                float(np.dot(query, ref) / (np.linalg.norm(query) * np.linalg.norm(ref) + 1e-8))
            )
        sims = np.asarray(sims, dtype=np.float64)
        conf = float(np.max(sims))
        weights = np.exp(self.temperature * (sims - sims.max()))
        weights = weights / np.maximum(weights.sum(), 1e-8)
        pred = (weights[:, None] * self.train_effects_[idxs]).sum(axis=0)
        unsafe = int(conf < self.unsafe_threshold)
        return pred.astype(np.float32), conf, unsafe

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        return np.stack([self._predict_one(tasks[int(i)])[0] for i in indices], axis=0)

    def predict_details(self, tasks: list[dict], indices: np.ndarray) -> dict:
        preds = []
        confs = []
        unsafes = []
        for i in indices:
            pred, conf, unsafe = self._predict_one(tasks[int(i)])
            preds.append(pred)
            confs.append(conf)
            unsafes.append(unsafe)
        import pandas as pd

        table = pd.DataFrame(
            {
                "transportability_score": np.asarray(confs, dtype=np.float64),
                "unsafe_flag": np.asarray(unsafes, dtype=int),
            }
        )
        return {
            "prediction": np.stack(preds, axis=0),
            "transportability": table,
        }


class V3UncertaintyTransport(V1ProgramTransport):
    name = "V3"

    def __init__(self, program_bank, alpha: float = 3.0, hash_dim: int = 64):
        super().__init__(program_bank, alpha=alpha, hash_dim=hash_dim)

    def uncertainty(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        selected = [tasks[int(i)] for i in indices]
        # Conservative proxy: sparse source support means unsafe transport.
        out = []
        train_mask = getattr(self, "train_mask_for_predict", np.ones(len(tasks), dtype=bool))
        for t in selected:
            support = sum(
                1
                for u, keep in zip(tasks, train_mask)
                if keep and u["perturbation"] == t["perturbation"] and u["context"] != t["context"]
            )
            out.append(1.0 / (1.0 + support))
        return np.asarray(out, dtype=np.float32)
