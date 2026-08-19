from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from encoders import pathway_prior_features
from evaluators import effect_metrics, pearson, rmse, spearman
from network_modules import NetworkModuleBank
from program_bank import ProgramBank
from transport_models import (
    ContextSimilarityBaseline,
    RidgeRegressor,
    V0StrongBaseline,
    V1ProgramTransport,
    V2GraphPriorTransport,
)


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _cosine_max(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q = _as_2d(query)
    b = _as_2d(bank)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return np.max(qn @ bn.T, axis=1)


class SafeTransPT:
    """Transportability-aware conservative perturbation-effect transport.

    The model keeps the strong mean-effect baseline, learns a raw program/graph
    transport map, then estimates whether that transport is safe for each target
    task.  The transportability score drives an adaptive blend:

        final = (1 - b) * baseline + b * transported

    Low-score tasks are not forced through transport; they receive only a tiny
    residual blend and are marked as unsafe for risk-coverage analysis.
    """

    name = "SafeTransPT"

    def __init__(
        self,
        n_programs: int = 128,
        seed: int = 0,
        bank_mode: str = "pca_nmf_hvg",
        min_blend: float = 0.02,
        max_blend: float = 0.18,
        unsafe_threshold: float = 0.42,
        no_abstain: bool = False,
        use_pathway: bool = True,
        learned_gate: bool = True,
        gate_cv_mode: str = "context",
        gate_folds: int = 4,
        gate_alpha: float = 2.0,
        gate_margin: float = 0.004,
    ):
        self.n_programs = int(n_programs)
        self.seed = int(seed)
        self.bank_mode = str(bank_mode)
        self.min_blend = float(min_blend)
        self.max_blend = float(max_blend)
        self.unsafe_threshold = float(unsafe_threshold)
        self.no_abstain = bool(no_abstain)
        self.use_pathway = bool(use_pathway)
        self.learned_gate = bool(learned_gate)
        self.gate_cv_mode = str(gate_cv_mode)
        self.gate_folds = int(gate_folds)
        self.gate_alpha = float(gate_alpha)
        self.gate_margin = float(gate_margin)

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "SafeTransPT":
        self.train_mask_for_predict = train_mask.copy()
        self.train_tasks_ = [t for t, keep in zip(tasks, train_mask) if keep]
        self.baseline_ = V0StrongBaseline().fit(tasks, train_mask)
        self.floor_ = V2GraphPriorTransport(
            ProgramBank(self.n_programs, self.seed, mode="pca_nmf_hvg"),
            alpha=5.0,
            hash_dim=96,
            blend=0.12,
        ).fit(tasks, train_mask)
        self.effect_scale_ = float(
            np.median([np.linalg.norm(t["effect"]) for t in self.train_tasks_])
        )
        self.effect_scale_ = max(self.effect_scale_, 1e-6)

        self.bank_ = ProgramBank(self.n_programs, self.seed, mode=self.bank_mode)
        if self.use_pathway:
            self.transport_ = V2GraphPriorTransport(
                self.bank_, alpha=5.0, hash_dim=96, blend=1.0
            ).fit(tasks, train_mask)
        else:
            self.transport_ = V1ProgramTransport(
                self.bank_, alpha=3.0, hash_dim=96
            ).fit(tasks, train_mask)

        self.train_effects_ = np.stack([t["effect"] for t in self.train_tasks_], axis=0)
        self.train_controls_ = np.stack([t["control_mean"] for t in self.train_tasks_], axis=0)
        self.train_prior_ = pathway_prior_features(
            [t["perturbation"] for t in self.train_tasks_],
            [t["context"] for t in self.train_tasks_],
            dim=96,
        )
        self.gate_reg_: RidgeRegressor | None = None
        self.by_pert_: dict[str, list[np.ndarray]] = {}
        for t in self.train_tasks_:
            self.by_pert_.setdefault(str(t["perturbation"]), []).append(t["effect"])
        if self.learned_gate:
            self._fit_gate(tasks, np.flatnonzero(train_mask))
        return self

    def _clone_for_gate(self) -> "SafeTransPT":
        kwargs = {
            "n_programs": self.n_programs,
            "seed": self.seed,
            "bank_mode": self.bank_mode,
            "min_blend": self.min_blend,
            "max_blend": self.max_blend,
            "unsafe_threshold": self.unsafe_threshold,
            "no_abstain": self.no_abstain,
            "use_pathway": self.use_pathway,
            "learned_gate": False,
            "gate_cv_mode": self.gate_cv_mode,
            "gate_folds": self.gate_folds,
            "gate_alpha": self.gate_alpha,
            "gate_margin": self.gate_margin,
        }
        if hasattr(self, "soft_power"):
            kwargs["soft_power"] = getattr(self, "soft_power")
        if hasattr(self, "max_network_genes"):
            kwargs["max_network_genes"] = getattr(self, "max_network_genes")
        if hasattr(self, "network_weight"):
            kwargs["network_weight"] = getattr(self, "network_weight")
        return self.__class__(**kwargs)  # type: ignore[call-arg]

    def _support_score(self, selected: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        support = []
        score = []
        for task in selected:
            vals = [
                t["effect"]
                for t in self.train_tasks_
                if t["perturbation"] == task["perturbation"] and t["context"] != task["context"]
            ]
            support.append(len(vals))
            score.append(1.0 - np.exp(-len(vals) / 2.0))
        return np.asarray(support, dtype=np.float64), np.asarray(score, dtype=np.float64)

    def _pert_consistency_score(self, selected: list[dict]) -> np.ndarray:
        out = []
        for task in selected:
            vals = self.by_pert_.get(str(task["perturbation"]), [])
            if len(vals) < 2:
                out.append(0.20)
                continue
            arr = np.stack(vals, axis=0)
            mean = arr.mean(axis=0)
            cors = [pearson(v, mean) for v in arr]
            cors = [c for c in cors if np.isfinite(c)]
            out.append(0.20 if not cors else float(np.clip((np.mean(cors) + 1.0) / 2.0, 0.0, 1.0)))
        return np.asarray(out, dtype=np.float64)

    def _pert_variance_score(self, selected: list[dict]) -> np.ndarray:
        out = []
        for task in selected:
            vals = self.by_pert_.get(str(task["perturbation"]), [])
            if len(vals) < 2:
                out.append(0.35)
                continue
            arr = np.stack(vals, axis=0)
            spread = float(np.mean(np.std(arr, axis=0))) / self.effect_scale_
            out.append(float(np.exp(-8.0 * spread)))
        return np.asarray(out, dtype=np.float64)

    def _feature_table(
        self,
        tasks: list[dict],
        indices: np.ndarray,
        baseline_pred: np.ndarray,
        transport_pred: np.ndarray,
    ) -> pd.DataFrame:
        feat_cols = [
            "support_score",
            "context_similarity",
            "perturbation_consistency",
            "perturbation_variance_score",
            "pathway_prior_similarity",
            "transport_baseline_disagreement",
            "disagreement_score",
        ]
        selected = [tasks[int(i)] for i in indices]
        support, support_score = self._support_score(selected)
        controls = np.stack([t["control_mean"] for t in selected], axis=0)
        ctx_sim = np.clip((_cosine_max(controls, self.train_controls_) + 1.0) / 2.0, 0.0, 1.0)
        query_prior = pathway_prior_features(
            [t["perturbation"] for t in selected],
            [t["context"] for t in selected],
            dim=96,
        )
        prior_sim = np.clip(_cosine_max(query_prior, self.train_prior_), 0.0, 1.0)
        consistency = self._pert_consistency_score(selected)
        variance = self._pert_variance_score(selected)
        disagreement = np.asarray(
            [rmse(baseline_pred[i], transport_pred[i]) for i in range(len(indices))],
            dtype=np.float64,
        )
        disagreement_score = np.exp(-2.0 * disagreement / self.effect_scale_)
        heuristic_score = (
            0.24 * support_score
            + 0.20 * ctx_sim
            + 0.18 * consistency
            + 0.16 * variance
            + 0.10 * prior_sim
            + 0.12 * disagreement_score
        )
        score = np.clip(heuristic_score, 0.0, 1.0)
        features = pd.DataFrame(
            {
                "support_score": support_score,
                "context_similarity": ctx_sim,
                "perturbation_consistency": consistency,
                "perturbation_variance_score": variance,
                "pathway_prior_similarity": prior_sim,
                "transport_baseline_disagreement": disagreement,
                "disagreement_score": disagreement_score,
            }
        )
        if self.gate_reg_ is not None:
            gate_x = features[feat_cols].to_numpy(dtype=np.float64)
            gate_score = 1.0 / (1.0 + np.exp(-self.gate_reg_.predict(gate_x)))
            score = np.clip(0.55 * score + 0.45 * gate_score, 0.0, 1.0)
        blend = self.min_blend + (self.max_blend - self.min_blend) * score
        unsafe = score < self.unsafe_threshold
        if not self.no_abstain:
            blend = np.where(unsafe, np.minimum(blend, self.min_blend), blend)
        features["transportability_score"] = score
        features["adaptive_blend"] = blend
        features["unsafe_flag"] = unsafe.astype(int)
        features["support_count"] = support
        return features[[
            "transportability_score",
            "adaptive_blend",
            "unsafe_flag",
            "support_count",
            "support_score",
            "context_similarity",
            "perturbation_consistency",
            "perturbation_variance_score",
            "pathway_prior_similarity",
            "transport_baseline_disagreement",
            "disagreement_score",
        ]]

    def _fit_gate(self, tasks: list[dict], train_indices: np.ndarray) -> None:
        if len(train_indices) < 8:
            return

        def _labels_and_features(model: "SafeTransPT", idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            floor_model = getattr(model, "floor_", None)
            if floor_model is None:
                floor_model = getattr(model, "fixed_v2_baseline_", None)
            if floor_model is None:
                return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)
            baseline_pred = floor_model.predict(tasks, idx)
            transport_pred = model.transport_.predict(tasks, idx)
            table = model._feature_table(tasks, idx, baseline_pred, transport_pred)
            y_true = np.stack([tasks[int(i)]["effect"] for i in idx], axis=0)
            labels = []
            for i in range(len(idx)):
                bank = getattr(model.transport_, "bank", None)
                if bank is not None:
                    true_prog = bank.transform(y_true[i : i + 1])[0]
                    base_prog = bank.transform(baseline_pred[i : i + 1])[0]
                    trans_prog = bank.transform(transport_pred[i : i + 1])[0]
                else:
                    true_prog = base_prog = trans_prog = None
                base_met = effect_metrics(y_true[i], baseline_pred[i], true_prog, base_prog)
                trans_met = effect_metrics(y_true[i], transport_pred[i], true_prog, trans_prog)
                improve_dims = sum(
                    [
                        trans_met["top20_overlap"] > base_met["top20_overlap"] + self.gate_margin,
                        trans_met["deg_precision_top50"] > base_met["deg_precision_top50"] + self.gate_margin,
                        trans_met["program_shift_consistency"] > base_met["program_shift_consistency"] + self.gate_margin,
                    ]
                )
                safe = int(improve_dims >= 2 and trans_met["rmse"] <= base_met["rmse"] + 0.02)
                labels.append(safe)
            x = table[[
                "support_score",
                "context_similarity",
                "perturbation_consistency",
                "perturbation_variance_score",
                "pathway_prior_similarity",
                "transport_baseline_disagreement",
                "disagreement_score",
            ]].to_numpy(dtype=np.float64)
            return x, np.asarray(labels, dtype=np.float64)

        feat_rows: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        group_key = "context" if self.gate_cv_mode != "perturbation" else "perturbation"
        group_values = sorted({str(tasks[int(i)][group_key]) for i in train_indices})
        if len(group_values) >= 2:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(group_values)
            folds = np.array_split(group_values, min(self.gate_folds, len(group_values)))
            for fold_values in folds:
                heldout = set(map(str, fold_values.tolist()))
                fold_test = np.asarray([int(i) for i in train_indices if str(tasks[int(i)][group_key]) in heldout], dtype=int)
                fold_train = np.asarray([int(i) for i in train_indices if str(tasks[int(i)][group_key]) not in heldout], dtype=int)
                if len(fold_train) < 8 or len(fold_test) < 2:
                    continue
                train_mask = np.zeros(len(tasks), dtype=bool)
                train_mask[fold_train] = True
                clone = self._clone_for_gate()
                clone.fit(tasks, train_mask)
                x, y = _labels_and_features(clone, fold_test)
                if len(x):
                    feat_rows.append(x)
                    label_rows.append(y)

        if feat_rows:
            x = np.concatenate(feat_rows, axis=0)
            y = np.concatenate(label_rows, axis=0)
        else:
            floor_model = getattr(self, "floor_", None)
            if floor_model is None:
                floor_model = getattr(self, "fixed_v2_baseline_", None)
            if floor_model is None:
                return
            baseline_pred = floor_model.predict(tasks, train_indices)
            transport_pred = self.transport_.predict(tasks, train_indices)
            table = self._feature_table(tasks, train_indices, baseline_pred, transport_pred)
            y_true = np.stack([tasks[int(i)]["effect"] for i in train_indices], axis=0)
            labels = []
            for i in range(len(train_indices)):
                bank = getattr(self.transport_, "bank", None)
                if bank is not None:
                    true_prog = bank.transform(y_true[i : i + 1])[0]
                    base_prog = bank.transform(baseline_pred[i : i + 1])[0]
                    trans_prog = bank.transform(transport_pred[i : i + 1])[0]
                else:
                    true_prog = base_prog = trans_prog = None
                base_met = effect_metrics(y_true[i], baseline_pred[i], true_prog, base_prog)
                trans_met = effect_metrics(y_true[i], transport_pred[i], true_prog, trans_prog)
                improve_dims = sum(
                    [
                        trans_met["top20_overlap"] > base_met["top20_overlap"] + self.gate_margin,
                        trans_met["deg_precision_top50"] > base_met["deg_precision_top50"] + self.gate_margin,
                        trans_met["program_shift_consistency"] > base_met["program_shift_consistency"] + self.gate_margin,
                    ]
                )
                safe = int(improve_dims >= 2 and trans_met["rmse"] <= base_met["rmse"] + 0.02)
                labels.append(safe)
            x = table[[
                "support_score",
                "context_similarity",
                "perturbation_consistency",
                "perturbation_variance_score",
                "pathway_prior_similarity",
                "transport_baseline_disagreement",
                "disagreement_score",
            ]].to_numpy(dtype=np.float64)
            y = np.asarray(labels, dtype=np.float64)

        if np.unique(y).size < 2:
            return
        self.gate_reg_ = RidgeRegressor(alpha=self.gate_alpha).fit(x, y)

    def predict_details(self, tasks: list[dict], indices: np.ndarray) -> dict[str, np.ndarray | pd.DataFrame]:
        v0_pred = self.baseline_.predict(tasks, indices)
        baseline_pred = self.floor_.predict(tasks, indices)
        transport_pred = self.transport_.predict(tasks, indices)
        table = self._feature_table(tasks, indices, baseline_pred, transport_pred)
        blend = table["adaptive_blend"].to_numpy(dtype=np.float32)
        pred = (baseline_pred + blend[:, None] * (transport_pred - baseline_pred)).astype(np.float32)
        return {
            "prediction": pred,
            "v0_prediction": v0_pred,
            "baseline_prediction": baseline_pred,
            "transport_prediction": transport_pred,
            "transportability": table,
        }

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        return self.predict_details(tasks, indices)["prediction"]  # type: ignore[index]

    def uncertainty(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        details = self.predict_details(tasks, indices)
        table = details["transportability"]  # type: ignore[assignment]
        return (1.0 - table["transportability_score"].to_numpy(dtype=np.float32)).astype(np.float32)


class SafeTransPTNoAbstain(SafeTransPT):
    name = "SafeTransPT_no_abstain"

    def __init__(self, *args, **kwargs):
        kwargs["no_abstain"] = True
        super().__init__(*args, **kwargs)


class SafeTransPTNoPathway(SafeTransPT):
    name = "SafeTransPT_no_pathway"

    def __init__(self, *args, **kwargs):
        kwargs["use_pathway"] = False
        super().__init__(*args, **kwargs)


class NetworkSafeTransPT(SafeTransPT):
    """Network-aware SafeTrans-PT using hdWGCNA-inspired modules.

    This variant is designed for the biological-explanation push.  It uses a
    co-expression module bank instead of a purely PCA/NMF bank, and it raises
    the transportability score only when source and target contexts are similar
    in module space.
    """

    name = "NetworkSafeTransPT"

    def __init__(
        self,
        n_programs: int = 96,
        seed: int = 0,
        soft_power: int = 6,
        max_network_genes: int = 1600,
        network_weight: float = 0.26,
        **kwargs,
    ):
        kwargs.setdefault("bank_mode", "network_modules")
        kwargs.setdefault("min_blend", 0.02)
        kwargs.setdefault("max_blend", 0.24)
        kwargs.setdefault("unsafe_threshold", 0.40)
        super().__init__(n_programs=n_programs, seed=seed, **kwargs)
        self.soft_power = int(soft_power)
        self.max_network_genes = int(max_network_genes)
        self.network_weight = float(network_weight)

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "NetworkSafeTransPT":
        self.train_mask_for_predict = train_mask.copy()
        self.train_tasks_ = [t for t, keep in zip(tasks, train_mask) if keep]
        self.baseline_ = V0StrongBaseline().fit(tasks, train_mask)
        self.effect_scale_ = float(np.median([np.linalg.norm(t["effect"]) for t in self.train_tasks_]))
        self.effect_scale_ = max(self.effect_scale_, 1e-6)
        self.gate_reg_ = None

        self.bank_ = NetworkModuleBank(
            n_modules=self.n_programs,
            seed=self.seed,
            soft_power=self.soft_power,
            max_network_genes=self.max_network_genes,
        )
        self.fixed_v2_baseline_ = V2GraphPriorTransport(
            ProgramBank(self.n_programs, self.seed, mode="pca_nmf_hvg"),
            alpha=5.0,
            hash_dim=96,
            blend=0.12,
        ).fit(tasks, train_mask)
        if self.use_pathway:
            self.transport_ = V2GraphPriorTransport(self.bank_, alpha=5.0, hash_dim=96, blend=1.0).fit(
                tasks, train_mask
            )
        else:
            self.transport_ = V1ProgramTransport(self.bank_, alpha=3.0, hash_dim=96).fit(tasks, train_mask)

        self.train_effects_ = np.stack([t["effect"] for t in self.train_tasks_], axis=0)
        self.train_controls_ = np.stack([t["control_mean"] for t in self.train_tasks_], axis=0)
        self.train_module_effects_ = self.bank_.transform(self.train_effects_)
        self.train_module_controls_ = self.bank_.transform(self.train_controls_)
        self.train_prior_ = pathway_prior_features(
            [t["perturbation"] for t in self.train_tasks_],
            [t["context"] for t in self.train_tasks_],
            dim=96,
        )
        self.by_pert_ = {}
        self.by_pert_module_ = {}
        for t, z in zip(self.train_tasks_, self.train_module_effects_):
            key = str(t["perturbation"])
            self.by_pert_.setdefault(key, []).append(t["effect"])
            self.by_pert_module_.setdefault(key, []).append(z)
        if self.learned_gate:
            self._fit_gate(tasks, np.flatnonzero(train_mask))
        return self

    def _network_preservation_scores(self, selected: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        controls = np.stack([t["control_mean"] for t in selected], axis=0)
        control_z = self.bank_.transform(controls)
        module_context_sim = np.clip((_cosine_max(control_z, self.train_module_controls_) + 1.0) / 2.0, 0.0, 1.0)
        module_consistency = []
        for task in selected:
            vals = self.by_pert_module_.get(str(task["perturbation"]), [])
            if len(vals) < 2:
                module_consistency.append(0.25)
                continue
            arr = np.stack(vals, axis=0)
            mean = arr.mean(axis=0)
            cors = [pearson(v, mean) for v in arr]
            cors = [c for c in cors if np.isfinite(c)]
            module_consistency.append(0.25 if not cors else float(np.clip((np.mean(cors) + 1.0) / 2.0, 0.0, 1.0)))
        return module_context_sim, np.asarray(module_consistency, dtype=np.float64)

    def _feature_table(
        self,
        tasks: list[dict],
        indices: np.ndarray,
        baseline_pred: np.ndarray,
        transport_pred: np.ndarray,
    ) -> pd.DataFrame:
        table = super()._feature_table(tasks, indices, baseline_pred, transport_pred)
        selected = [tasks[int(i)] for i in indices]
        module_context_sim, module_consistency = self._network_preservation_scores(selected)
        old_score = table["transportability_score"].to_numpy(dtype=np.float64)
        network_score = 0.60 * module_context_sim + 0.40 * module_consistency
        w = float(np.clip(self.network_weight, 0.0, 0.8))
        score = np.clip((1.0 - w) * old_score + w * network_score, 0.0, 1.0)
        blend = self.min_blend + (self.max_blend - self.min_blend) * score
        unsafe = score < self.unsafe_threshold
        if not self.no_abstain:
            blend = np.where(unsafe, np.minimum(blend, self.min_blend), blend)
        table["network_module_context_similarity"] = module_context_sim
        table["network_module_perturbation_consistency"] = module_consistency
        table["network_preservation_score"] = network_score
        table["transportability_score"] = score
        table["adaptive_blend"] = blend
        table["unsafe_flag"] = unsafe.astype(int)
        return table

    def predict_details(self, tasks: list[dict], indices: np.ndarray) -> dict[str, np.ndarray | pd.DataFrame]:
        # Use the fixed V2 transport as the floor.  The network-aware module
        # branch is allowed to add a residual only when module preservation says
        # the transport is plausible.  This makes the hdWGCNA-inspired branch a
        # conservative improvement attempt rather than a wholesale replacement.
        baseline_pred = self.fixed_v2_baseline_.predict(tasks, indices)
        transport_pred = self.transport_.predict(tasks, indices)
        table = self._feature_table(tasks, indices, baseline_pred, transport_pred)
        blend = table["adaptive_blend"].to_numpy(dtype=np.float32)
        pred = (baseline_pred + blend[:, None] * (transport_pred - baseline_pred)).astype(np.float32)
        return {
            "prediction": pred,
            "baseline_prediction": baseline_pred,
            "transport_prediction": transport_pred,
            "transportability": table,
        }


class NetworkSafeTransPTNoAbstain(NetworkSafeTransPT):
    name = "NetworkSafeTransPT_no_abstain"

    def __init__(self, *args, **kwargs):
        kwargs["no_abstain"] = True
        super().__init__(*args, **kwargs)


def _rank_lower_is_better(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks


def _rank_higher_is_better(values: np.ndarray) -> np.ndarray:
    return _rank_lower_is_better(-np.asarray(values, dtype=np.float64))


def _row_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return arr / norm


class PolicySafeTransPT:
    """Expert router for perturbation transport.

    The idea is simpler than the earlier gate:
    we let several candidate predictors compete and learn which one to trust
    per task.  That is closer to the real problem than a single yes/no safety
    switch.
    """

    name = "PolicySafeTransPT"

    def __init__(
        self,
        n_programs: int = 128,
        seed: int = 0,
        bank_mode: str = "pca_nmf_hvg",
        policy_folds: int = 3,
        confidence_threshold: float = 0.45,
        rank_margin: float = 0.75,
        max_network_genes: int = 1600,
        retrieval_k: int = 5,
        router_weight: float = 0.65,
        retrieval_weight: float = 0.35,
        calibrate_threshold: bool = True,
        abstain_quantile: float = 0.15,
        routing_mode: str = "hard",
        utility_margin: float = 0.004,
        risk_alpha: float = 4.0,
        fallback_on_unsafe: bool = False,
        rank_graft_strength: float = 0.55,
        rank_graft_mass_threshold: float = 0.42,
    ):
        self.n_programs = int(n_programs)
        self.seed = int(seed)
        self.bank_mode = str(bank_mode)
        self.policy_folds = int(policy_folds)
        self.confidence_threshold = float(confidence_threshold)
        self.rank_margin = float(rank_margin)
        self.max_network_genes = int(max_network_genes)
        self.retrieval_k = int(retrieval_k)
        self.router_weight = float(router_weight)
        self.retrieval_weight = float(retrieval_weight)
        self.calibrate_threshold = bool(calibrate_threshold)
        self.abstain_quantile = float(abstain_quantile)
        self.routing_mode = str(routing_mode)
        self.utility_margin = float(utility_margin)
        self.risk_alpha = float(risk_alpha)
        self.fallback_on_unsafe = bool(fallback_on_unsafe)
        self.rank_graft_strength = float(rank_graft_strength)
        self.rank_graft_mass_threshold = float(rank_graft_mass_threshold)

    def _expert_names(self) -> list[str]:
        return ["V0", "V1", "V2", "Safe", "Network", "ContextSim"]

    def _state_from_indices(self, tasks: list[dict], indices: np.ndarray) -> dict:
        train_tasks = [tasks[int(i)] for i in indices]
        train_effects = np.stack([t["effect"] for t in train_tasks], axis=0)
        train_controls = np.stack([t["control_mean"] for t in train_tasks], axis=0)
        train_prior = pathway_prior_features(
            [t["perturbation"] for t in train_tasks],
            [t["context"] for t in train_tasks],
            dim=96,
        )
        effect_scale = float(np.median([np.linalg.norm(t["effect"]) for t in train_tasks]))
        effect_scale = max(effect_scale, 1e-6)
        return {
            "train_tasks": train_tasks,
            "train_effects": train_effects,
            "train_controls": train_controls,
            "train_prior": train_prior,
            "train_retrieval_repr": np.concatenate([train_controls, train_prior], axis=1),
            "effect_scale": effect_scale,
        }

    def _fit_experts(self, tasks: list[dict], train_mask: np.ndarray) -> dict[str, object]:
        train_effects = np.stack([t["effect"] for t, keep in zip(tasks, train_mask) if keep], axis=0)
        experts: dict[str, object] = {
            "V0": V0StrongBaseline().fit(tasks, train_mask),
            "V1": V1ProgramTransport(ProgramBank(self.n_programs, self.seed, mode="pca"), alpha=3.0).fit(tasks, train_mask),
            "V2": V2GraphPriorTransport(
                ProgramBank(self.n_programs, self.seed, mode=self.bank_mode), alpha=5.0, blend=0.12
            ).fit(tasks, train_mask),
            "Safe": SafeTransPT(
                n_programs=self.n_programs,
                seed=self.seed,
                bank_mode=self.bank_mode,
                max_blend=0.24,
                learned_gate=False,
            ).fit(tasks, train_mask),
            "Network": NetworkSafeTransPTNoAbstain(
                n_programs=max(64, self.n_programs // 2),
                seed=self.seed,
                max_network_genes=self.max_network_genes,
                learned_gate=False,
            ).fit(tasks, train_mask),
            "ContextSim": ContextSimilarityBaseline().fit(tasks, train_mask),
        }
        # keep a compact bank for router feature computations
        experts["_router_bank"] = ProgramBank(self.n_programs, self.seed, mode=self.bank_mode).fit(train_effects)
        return experts

    def _make_folds(self, tasks: list[dict], indices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(self.seed)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for group_key in ["context", "perturbation"]:
            groups = sorted({str(tasks[int(i)][group_key]) for i in indices})
            if len(groups) < 2:
                continue
            rng.shuffle(groups)
            n_folds = min(self.policy_folds, len(groups))
            for chunk in np.array_split(groups, n_folds):
                heldout = set(map(str, chunk.tolist()))
                test = np.asarray([int(i) for i in indices if str(tasks[int(i)][group_key]) in heldout], dtype=int)
                train = np.asarray([int(i) for i in indices if str(tasks[int(i)][group_key]) not in heldout], dtype=int)
                if len(train) >= 4 and len(test) >= 2:
                    folds.append((train, test))
        return folds

    def _predict_experts(
        self,
        tasks: list[dict],
        experts: dict[str, object],
        indices: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        preds: dict[str, np.ndarray] = {}
        confs: dict[str, np.ndarray] = {}
        for name in self._expert_names():
            model = experts[name]
            if hasattr(model, "predict_details"):
                details = model.predict_details(tasks, indices)
                preds[name] = details["prediction"]
                table = details["transportability"]
                if isinstance(table, pd.DataFrame) and "transportability_score" in table:
                    confs[name] = table["transportability_score"].to_numpy(dtype=np.float64)
                else:
                    confs[name] = np.ones(len(indices), dtype=np.float64)
            else:
                preds[name] = model.predict(tasks, indices)  # type: ignore[assignment]
                confs[name] = np.ones(len(indices), dtype=np.float64)
        return preds, confs

    def _router_features(
        self,
        tasks: list[dict],
        indices: np.ndarray,
        train_state: dict,
        preds: dict[str, np.ndarray],
        confs: dict[str, np.ndarray],
    ) -> np.ndarray:
        selected = [tasks[int(i)] for i in indices]
        train_tasks = train_state["train_tasks"]
        support = []
        support_score = []
        for task in selected:
            vals = [
                t["effect"]
                for t in train_tasks
                if t["perturbation"] == task["perturbation"] and t["context"] != task["context"]
            ]
            support.append(len(vals))
            support_score.append(1.0 - np.exp(-len(vals) / 2.0))
        controls = np.stack([t["control_mean"] for t in selected], axis=0)
        ctx_sim = np.clip((_cosine_max(controls, train_state["train_controls"]) + 1.0) / 2.0, 0.0, 1.0)

        consistency = []
        variance = []
        for task in selected:
            vals = [t["effect"] for t in train_tasks if t["perturbation"] == task["perturbation"]]
            if len(vals) < 2:
                consistency.append(0.20)
                variance.append(0.35)
                continue
            arr = np.stack(vals, axis=0)
            mean = arr.mean(axis=0)
            cors = [pearson(v, mean) for v in arr]
            cors = [c for c in cors if np.isfinite(c)]
            consistency.append(0.20 if not cors else float(np.clip((np.mean(cors) + 1.0) / 2.0, 0.0, 1.0)))
            spread = float(np.mean(np.std(arr, axis=0))) / train_state["effect_scale"]
            variance.append(float(np.exp(-8.0 * spread)))
        prior = pathway_prior_features(
            [t["perturbation"] for t in selected],
            [t["context"] for t in selected],
            dim=96,
        )
        prior_sim = np.clip(_cosine_max(prior, train_state["train_prior"]), 0.0, 1.0)

        expert_rmse = []
        expert_cos = []
        expert_norms = []
        expert_confs = []
        expert_names = self._expert_names()
        for name in expert_names:
            pred = preds[name]
            expert_norms.append(np.linalg.norm(pred, axis=1))
            expert_confs.append(confs[name])
        for i, a in enumerate(expert_names):
            for b in expert_names[i + 1 :]:
                pa = preds[a]
                pb = preds[b]
                expert_rmse.append(np.sqrt(np.mean((pa - pb) ** 2, axis=1)))
                num = np.sum(pa * pb, axis=1)
                den = np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1) + 1e-8
                expert_cos.append(num / den)
        expert_rmse = np.stack(expert_rmse, axis=1)
        expert_cos = np.stack(expert_cos, axis=1)
        expert_norms = np.stack(expert_norms, axis=1)
        expert_confs = np.stack(expert_confs, axis=1)

        feat = np.column_stack(
            [
                np.asarray(support, dtype=np.float64),
                np.asarray(support_score, dtype=np.float64),
                ctx_sim,
                np.asarray(consistency, dtype=np.float64),
                np.asarray(variance, dtype=np.float64),
                prior_sim,
                expert_rmse.mean(axis=1),
                expert_rmse.max(axis=1),
                expert_rmse.std(axis=1),
                expert_cos.mean(axis=1),
                expert_cos.min(axis=1),
                expert_norms.mean(axis=1),
                expert_norms.std(axis=1),
                expert_confs.mean(axis=1),
                expert_confs.max(axis=1),
                expert_confs.min(axis=1),
                np.abs(expert_confs[:, 2] - expert_confs[:, 0]),
                np.abs(expert_confs[:, 3] - expert_confs[:, 1]),
            ]
        ).astype(np.float32)
        return feat

    def _metric_targets_from_predictions(
        self,
        y_true: np.ndarray,
        preds: dict[str, np.ndarray],
        bank: ProgramBank,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        expert_names = self._expert_names()
        all_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in expert_names}
        for i in range(len(y_true)):
            true_prog = bank.transform(y_true[i : i + 1])[0]
            for name in expert_names:
                pred = preds[name][i]
                pred_prog = bank.transform(pred[None, :])[0]
                all_metrics[name].append(effect_metrics(y_true[i], pred, true_prog, pred_prog))
        labels = []
        utilities = []
        errors = []
        for i in range(len(y_true)):
            top20 = np.asarray([all_metrics[name][i]["top20_overlap"] for name in expert_names], dtype=np.float64)
            deg = np.asarray([all_metrics[name][i]["deg_precision_top50"] for name in expert_names], dtype=np.float64)
            prog = np.asarray([all_metrics[name][i]["program_shift_consistency"] for name in expert_names], dtype=np.float64)
            err = np.asarray([all_metrics[name][i]["rmse"] for name in expert_names], dtype=np.float64)
            corr = np.asarray([all_metrics[name][i]["pearson"] for name in expert_names], dtype=np.float64)
            spear = np.asarray([all_metrics[name][i]["spearman"] for name in expert_names], dtype=np.float64)
            err_scale = max(float(np.nanmedian(err)), 1e-6)
            err_penalty = err / err_scale
            utility = (
                0.31 * np.nan_to_num(top20, nan=0.0)
                + 0.31 * np.nan_to_num(deg, nan=0.0)
                + 0.22 * np.nan_to_num(prog, nan=0.0)
                + 0.08 * np.nan_to_num(corr, nan=0.0)
                + 0.04 * np.nan_to_num(spear, nan=0.0)
                - 0.04 * err_penalty
            )
            best = int(np.argmax(utility))
            baseline = 0
            if best != baseline and (utility[best] - utility[baseline]) < self.utility_margin:
                best = baseline
            labels.append(best)
            utilities.append(utility)
            errors.append(err)
        return np.asarray(labels, dtype=int), np.asarray(utilities, dtype=np.float64), np.asarray(errors, dtype=np.float64)

    def _labels_from_metrics(
        self,
        y_true: np.ndarray,
        preds: dict[str, np.ndarray],
        bank: ProgramBank,
    ) -> np.ndarray:
        labels, _, _ = self._metric_targets_from_predictions(y_true, preds, bank)
        return labels

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "PolicySafeTransPT":
        self.train_mask_for_predict = train_mask.copy()
        train_indices = np.flatnonzero(train_mask)
        self.train_state_ = self._state_from_indices(tasks, train_indices)

        X_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        utility_rows: list[np.ndarray] = []
        error_rows: list[np.ndarray] = []
        vote_map: dict[int, list[int]] = {int(i): [] for i in train_indices}
        for fold_train, fold_test in self._make_folds(tasks, train_indices):
            fold_mask = np.zeros(len(tasks), dtype=bool)
            fold_mask[fold_train] = True
            fold_state = self._state_from_indices(tasks, fold_train)
            experts = self._fit_experts(tasks, fold_mask)
            preds, confs = self._predict_experts(tasks, experts, fold_test)
            router_x = self._router_features(tasks, fold_test, fold_state, preds, confs)
            bank = experts["_router_bank"]
            y_true = np.stack([tasks[int(i)]["effect"] for i in fold_test], axis=0)
            y, utility_y, error_y = self._metric_targets_from_predictions(y_true, preds, bank)  # type: ignore[arg-type]
            X_rows.append(router_x)
            y_rows.append(y)
            utility_rows.append(utility_y)
            error_rows.append(error_y)
            for idx, label in zip(fold_test.tolist(), y.tolist()):
                vote_map.setdefault(int(idx), []).append(int(label))

        if X_rows:
            X = np.vstack(X_rows)
            y = np.concatenate(y_rows)
            utility_y = np.vstack(utility_rows)
            error_y = np.vstack(error_rows)
            self.utility_reg_ = RidgeRegressor(alpha=self.risk_alpha).fit(X, utility_y)
            self.error_reg_ = RidgeRegressor(alpha=self.risk_alpha).fit(X, error_y)
            self.risk_error_scale_ = float(max(np.nanmedian(error_y), 1e-6))
            if np.unique(y).size >= 2:
                self.router_ = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=self.seed,
                    ),
                )
                self.router_.fit(X, y)
            else:
                self.router_ = None
        else:
            self.router_ = None
            self.utility_reg_ = None
            self.error_reg_ = None
            self.risk_error_scale_ = float(self.train_state_["effect_scale"])

        self.expert_order_ = self._expert_names()
        self.retrieval_scaler_ = StandardScaler().fit(self.train_state_["train_retrieval_repr"])
        retr_scaled = self.retrieval_scaler_.transform(self.train_state_["train_retrieval_repr"])
        self.retrieval_nn_ = None
        if len(retr_scaled) >= 2:
            self.retrieval_nn_ = NearestNeighbors(
                n_neighbors=min(max(1, self.retrieval_k), len(retr_scaled)),
                metric="euclidean",
            ).fit(retr_scaled)
        self.train_label_map_ = np.zeros(len(train_indices), dtype=int)
        self.train_index_lookup_ = {int(idx): pos for pos, idx in enumerate(train_indices.tolist())}
        for pos, idx in enumerate(train_indices.tolist()):
            votes = vote_map.get(int(idx), [])
            if votes:
                self.train_label_map_[pos] = int(np.bincount(votes, minlength=len(self.expert_order_)).argmax())
        if self.calibrate_threshold and X_rows:
            try:
                pred_util = self.utility_reg_.predict(X) if self.utility_reg_ is not None else utility_y
                pred_err = self.error_reg_.predict(X) if self.error_reg_ is not None else error_y
                selected = np.argmax(pred_util, axis=1)
                margin = pred_util[np.arange(len(selected)), selected] - pred_util[:, 0]
                selected = np.where((selected != 0) & (margin < self.utility_margin), 0, selected)
                conf = self._calibrated_confidence(
                    X,
                    selected,
                    pred_util,
                    pred_err,
                    full_probs=None,
                    retrieval_conf=None,
                )
                q = float(np.quantile(conf, np.clip(self.abstain_quantile, 0.02, 0.45)))
                self.confidence_threshold_ = float(np.clip(q, 0.05, 0.95))
            except Exception:
                self.confidence_threshold_ = float(np.clip(self.confidence_threshold, 0.05, 0.99))
        else:
            self.confidence_threshold_ = float(np.clip(self.confidence_threshold, 0.05, 0.99))

        self.full_experts_ = self._fit_experts(tasks, train_mask)
        return self

    def _finalize_probs(self, probs: np.ndarray) -> np.ndarray:
        full = np.zeros((probs.shape[0], len(self.expert_order_)), dtype=np.float64)
        if getattr(self, "router_", None) is None:
            full[:, 0] = 1.0
            return full
        classes = list(self.router_.named_steps["logisticregression"].classes_)  # type: ignore[index]
        class_to_col = {int(c): i for i, c in enumerate(classes)}
        for j in range(len(self.expert_order_)):
            if j in class_to_col:
                full[:, j] = probs[:, class_to_col[j]]
        row_sum = full.sum(axis=1, keepdims=True)
        row_sum[row_sum < 1e-8] = 1.0
        return full / row_sum

    def _retrieval_prior(self, tasks: list[dict], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if getattr(self, "retrieval_nn_", None) is None:
            prior = np.zeros((len(indices), len(self.expert_order_)), dtype=np.float64)
            prior[:, 0] = 1.0
            return prior, np.ones(len(indices), dtype=np.float64), np.zeros(len(indices), dtype=np.float64)
        selected = [tasks[int(i)] for i in indices]
        query = np.concatenate(
            [
                np.stack([t["control_mean"] for t in selected], axis=0),
                pathway_prior_features(
                    [t["perturbation"] for t in selected],
                    [t["context"] for t in selected],
                    dim=96,
                ),
            ],
            axis=1,
        )
        query = self.retrieval_scaler_.transform(query)
        dist, neigh = self.retrieval_nn_.kneighbors(query, return_distance=True)
        prior = np.zeros((len(indices), len(self.expert_order_)), dtype=np.float64)
        conf = np.zeros(len(indices), dtype=np.float64)
        entropy = np.zeros(len(indices), dtype=np.float64)
        weights = np.exp(-2.5 * dist)
        for i in range(len(indices)):
            label_idx = np.asarray([self.train_label_map_[j] for j in neigh[i]], dtype=int)
            w = weights[i]
            vote = np.bincount(label_idx, weights=w, minlength=len(self.expert_order_)).astype(np.float64)
            if vote.sum() <= 1e-8:
                vote[0] = 1.0
            prob = vote / vote.sum()
            prior[i] = prob
            conf[i] = float(np.max(prob))
            entropy[i] = float(-np.sum(prob * np.log(prob + 1e-8)))
        return prior, conf, entropy

    def _calibrated_confidence(
        self,
        router_x: np.ndarray,
        selected: np.ndarray,
        pred_util: np.ndarray | None,
        pred_err: np.ndarray | None,
        full_probs: np.ndarray | None,
        retrieval_conf: np.ndarray | None,
    ) -> np.ndarray:
        x = np.asarray(router_x, dtype=np.float64)
        support_score = np.clip(x[:, 1], 0.0, 1.0)
        context_similarity = np.clip(x[:, 2], 0.0, 1.0)
        consistency = np.clip(x[:, 3], 0.0, 1.0)
        variance = np.clip(x[:, 4], 0.0, 1.0)
        prior_similarity = np.clip(x[:, 5], 0.0, 1.0)
        disagreement = np.maximum(x[:, 6], 0.0)
        cos_agreement = np.clip((x[:, 9] + 1.0) / 2.0, 0.0, 1.0)
        structural = (
            0.24 * support_score
            + 0.24 * context_similarity
            + 0.18 * consistency
            + 0.14 * variance
            + 0.12 * prior_similarity
            + 0.08 * cos_agreement
        )
        disagreement_score = np.exp(-disagreement / max(float(self.train_state_.get("effect_scale", 1.0)), 1e-6))
        if pred_err is None:
            err_score = disagreement_score
        else:
            err = np.asarray(pred_err, dtype=np.float64)
            row = np.arange(len(selected))
            chosen_err = np.maximum(err[row, selected], 0.0)
            scale = max(float(getattr(self, "risk_error_scale_", np.nanmedian(chosen_err))), 1e-6)
            err_score = np.exp(-chosen_err / scale)
        if pred_util is None:
            margin_score = np.ones(len(selected), dtype=np.float64) * 0.5
        else:
            util = np.asarray(pred_util, dtype=np.float64)
            row = np.arange(len(selected))
            margin = util[row, selected] - util[:, 0]
            margin_score = 1.0 / (1.0 + np.exp(-margin / max(abs(self.utility_margin), 1e-3)))
            margin_score = np.where(selected == 0, np.maximum(margin_score, 0.62), margin_score)
        if full_probs is None:
            prob_score = np.ones(len(selected), dtype=np.float64) * 0.5
        else:
            prob_score = np.max(np.asarray(full_probs, dtype=np.float64), axis=1)
        if retrieval_conf is None:
            retrieval_score = prob_score
        else:
            retrieval_score = np.clip(np.asarray(retrieval_conf, dtype=np.float64), 0.0, 1.0)
        score = (
            0.34 * err_score
            + 0.24 * structural
            + 0.16 * disagreement_score
            + 0.14 * margin_score
            + 0.07 * retrieval_score
            + 0.05 * prob_score
        )
        return np.clip(score, 0.0, 1.0)

    def _rank_preserving_graft(
        self,
        base_pred: np.ndarray,
        pred: np.ndarray,
        preds: dict[str, np.ndarray],
        transport_mass: np.ndarray,
    ) -> np.ndarray:
        if self.rank_graft_strength <= 0:
            return pred
        support_names = [name for name in ["V2", "Safe", "Network", "ContextSim"] if name in preds]
        if not support_names:
            return pred
        out = np.asarray(pred, dtype=np.float32).copy()
        support_stack = np.stack([preds[name] for name in support_names], axis=0).astype(np.float64)
        support_mag = np.mean(np.abs(support_stack), axis=0)
        support_signed = np.mean(support_stack, axis=0)
        base = np.asarray(base_pred, dtype=np.float64)
        n_genes = base.shape[1]
        keep_k = min(20, n_genes)
        graft_k = min(50, n_genes)
        for i in range(base.shape[0]):
            if transport_mass[i] < self.rank_graft_mass_threshold:
                continue
            base_abs = np.abs(base[i])
            preserve = set(np.argsort(-base_abs)[:keep_k].tolist())
            pool = np.argsort(-support_mag[i])[:graft_k]
            if keep_k < n_genes:
                upper = float(np.sort(base_abs)[-keep_k])
            else:
                upper = float(base_abs.max())
            if graft_k < n_genes:
                lower = float(np.sort(base_abs)[-graft_k])
            else:
                lower = float(np.median(base_abs))
            cap = max(lower, 0.985 * upper)
            for j in pool:
                jj = int(j)
                if jj in preserve:
                    continue
                target_mag = min(max(float(support_mag[i, jj]), float(base_abs[jj])), cap)
                if target_mag <= base_abs[jj] + 1e-8:
                    continue
                sign = np.sign(support_signed[i, jj])
                if sign == 0:
                    sign = np.sign(base[i, jj]) if base[i, jj] != 0 else 1.0
                mixed_mag = (1.0 - self.rank_graft_strength) * base_abs[jj] + self.rank_graft_strength * target_mag
                out[i, jj] = float(sign * mixed_mag)
        return out

    def predict_details(self, tasks: list[dict], indices: np.ndarray) -> dict[str, np.ndarray | pd.DataFrame]:
        experts = self.full_experts_
        preds, confs = self._predict_experts(tasks, experts, indices)
        router_x = self._router_features(tasks, indices, self.train_state_, preds, confs)
        if getattr(self, "router_", None) is None:
            full_probs = np.zeros((len(indices), len(self.expert_order_)), dtype=np.float64)
            full_probs[:, 0] = 1.0
        else:
            raw_probs = self.router_.predict_proba(router_x)  # type: ignore[union-attr]
            full_probs = self._finalize_probs(raw_probs)
        router_confidence = np.max(full_probs, axis=1)
        retrieval_probs, retrieval_conf, retrieval_entropy = self._retrieval_prior(tasks, indices)
        full_probs = self.router_weight * full_probs + self.retrieval_weight * retrieval_probs
        full_probs = full_probs / np.maximum(full_probs.sum(axis=1, keepdims=True), 1e-8)
        expert_stack = np.stack([preds[name] for name in self.expert_order_], axis=0)
        pred_util = None
        pred_err = None
        if getattr(self, "utility_reg_", None) is not None:
            pred_util = self.utility_reg_.predict(router_x)
            prob_nudge = 0.015 * (full_probs - full_probs[:, [0]])
            pred_util = pred_util + prob_nudge
            selected = np.argmax(pred_util, axis=1)
            margin = pred_util[np.arange(len(selected)), selected] - pred_util[:, 0]
            selected = np.where((selected != 0) & (margin < self.utility_margin), 0, selected)
        else:
            selected = np.argmax(full_probs, axis=1)
        if getattr(self, "error_reg_", None) is not None:
            pred_err = np.maximum(self.error_reg_.predict(router_x), 0.0)
        if self.routing_mode == "soft":
            weights = full_probs[:, :, None]
            pred = np.sum(weights * expert_stack.transpose(1, 0, 2), axis=1).astype(np.float32)
        elif self.routing_mode == "hybrid":
            selected_pred = expert_stack.transpose(1, 0, 2)[np.arange(len(indices)), selected]
            baseline_pred = preds["V0"]
            selected_conf = np.max(full_probs, axis=1)
            pred = (baseline_pred + selected_conf[:, None] * (selected_pred - baseline_pred)).astype(np.float32)
        else:
            pred = expert_stack.transpose(1, 0, 2)[np.arange(len(indices)), selected].astype(np.float32)
        transport_mass = 1.0 - full_probs[:, 0]
        pred = self._rank_preserving_graft(preds["V0"], pred, preds, transport_mass)
        max_prob = self._calibrated_confidence(
            router_x,
            selected,
            pred_util,
            pred_err,
            full_probs,
            retrieval_conf,
        )
        baseline_prob = full_probs[:, 0]
        threshold = getattr(self, "confidence_threshold_", self.confidence_threshold)
        if len(max_prob) >= 5:
            threshold = max(float(threshold), float(np.quantile(max_prob, np.clip(self.abstain_quantile, 0.05, 0.35))))
        unsafe = (max_prob <= threshold).astype(int)
        if self.fallback_on_unsafe:
            pred[unsafe.astype(bool)] = preds["V0"][unsafe.astype(bool)]
        table_payload = {
            "transportability_score": max_prob,
            "adaptive_blend": 1.0 - baseline_prob,
            "unsafe_flag": unsafe,
            "selected_expert": [self.expert_order_[i] for i in selected],
            "entropy": -np.sum(full_probs * np.log(full_probs + 1e-8), axis=1),
            "router_confidence": router_confidence,
            "retrieval_confidence": retrieval_conf,
            "retrieval_entropy": retrieval_entropy,
        }
        for j, name in enumerate(self.expert_order_):
            table_payload[f"p_{name}"] = full_probs[:, j]
        if pred_err is not None:
            row = np.arange(len(selected))
            table_payload["predicted_selected_rmse"] = pred_err[row, selected]
            table_payload["predicted_v0_rmse"] = pred_err[:, 0]
        if pred_util is not None:
            row = np.arange(len(selected))
            table_payload["predicted_selected_utility"] = pred_util[row, selected]
            table_payload["predicted_v0_utility"] = pred_util[:, 0]
        table = pd.DataFrame(table_payload)
        table["support_count"] = router_x[:, 0]
        table["support_score"] = router_x[:, 1]
        table["context_similarity"] = router_x[:, 2]
        table["perturbation_consistency"] = router_x[:, 3]
        table["perturbation_variance_score"] = router_x[:, 4]
        table["pathway_prior_similarity"] = router_x[:, 5]
        return {
            "prediction": pred,
            "transportability": table,
        }

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        return self.predict_details(tasks, indices)["prediction"]  # type: ignore[index]

    def uncertainty(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        details = self.predict_details(tasks, indices)
        table = details["transportability"]  # type: ignore[assignment]
        return (1.0 - table["transportability_score"].to_numpy(dtype=np.float32)).astype(np.float32)
