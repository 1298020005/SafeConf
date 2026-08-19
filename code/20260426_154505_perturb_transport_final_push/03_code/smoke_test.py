from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluators import effect_metrics
from program_bank import ProgramBank
from transport_models import V0StrongBaseline, V1ProgramTransport


def main() -> None:
    rng = np.random.default_rng(7)
    tasks = []
    for c in ["ctxA", "ctxB", "ctxC"]:
        ctrl = rng.normal(size=40).astype(np.float32)
        for p in ["P1", "P2", "P3"]:
            base = rng.normal(size=40).astype(np.float32)
            effect = base + (0.2 if c == "ctxC" else 0.0)
            tasks.append({"context": c, "perturbation": p, "control_mean": ctrl, "effect": effect, "dataset": "smoke"})
    train_mask = np.array([t["context"] != "ctxC" for t in tasks])
    test_idx = np.array([i for i, t in enumerate(tasks) if t["context"] == "ctxC"], dtype=int)
    bank = ProgramBank(8).fit(np.stack([t["effect"] for t, keep in zip(tasks, train_mask) if keep]))
    models = [V0StrongBaseline().fit(tasks, train_mask), V1ProgramTransport(ProgramBank(8)).fit(tasks, train_mask)]
    for model in models:
        if hasattr(model, "train_mask_for_predict"):
            model.train_mask_for_predict = train_mask
        pred = model.predict(tasks, test_idx)
        true = np.stack([tasks[i]["effect"] for i in test_idx])
        zt = bank.transform(true)
        zp = bank.transform(pred)
        m = effect_metrics(true[0], pred[0], zt[0], zp[0])
        assert "pearson" in m and pred.shape == true.shape
    print("smoke_test_passed")


if __name__ == "__main__":
    main()
