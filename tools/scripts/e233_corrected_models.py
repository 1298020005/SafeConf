"""E233 models that repair a frozen PerturBench train/predict contract mismatch."""
from __future__ import annotations

import torch.nn.functional as F

from perturbench.modelcore.models.linear_additive import LinearAdditive
from perturbench.data.types import Batch


class CorrectedLinearAdditive(LinearAdditive):
    """LinearAdditive trained from matched controls instead of treated inputs."""

    def _shared_step(self, batch: Batch, name: str):
        predicted = self.forward(
            batch.controls.squeeze(), batch.perturbations.squeeze(), batch.covariates
        )
        loss = F.mse_loss(predicted, batch.gene_expression.squeeze())
        self.log(
            name,
            loss,
            on_step=name.startswith("val"),
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: Batch, batch_idx: int):
        if batch.controls is None:
            raise RuntimeError("E233 requires matched controls")
        return self._shared_step(batch, "train_loss")

    def validation_step(self, batch: Batch, batch_idx: int):
        if batch.controls is None:
            raise RuntimeError("E233 requires matched controls")
        return self._shared_step(batch, "val_loss")
