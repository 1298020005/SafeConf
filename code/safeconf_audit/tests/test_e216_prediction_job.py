import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "scripts"
    / "run_e216_prediction_job.py"
)
SPEC = importlib.util.spec_from_file_location("run_e216_prediction_job", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prediction_frame_is_unique_and_stably_sorted(tmp_path):
    path = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "condition": ["B", "A"],
            "cell_type": ["z", "a"],
            "treatment": ["x", "x"],
            "ignored": [1, 2],
        }
    ).to_csv(path, index=False)
    frame = MODULE.load_prediction_frame(path, require_224=False)
    assert list(frame.columns) == ["condition", "cell_type", "treatment"]
    assert frame.condition.tolist() == ["A", "B"]


def test_choose_checkpoint_requires_run_lineage(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    status = {"last_checkpoint": str(checkpoint)}
    assert MODULE.choose_checkpoint(run, status, None) == checkpoint.resolve()

    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(b"other")
    status["last_checkpoint"] = str(outside)
    try:
        MODULE.choose_checkpoint(run, status, None)
    except ValueError as exc:
        assert "declared training run" in str(exc)
    else:
        raise AssertionError("outside checkpoint must be rejected")


def test_aggregate_prediction_chunks_preserves_registered_order(tmp_path):
    first = ad.AnnData(
        X=np.asarray([[1.0, 3.0], [3.0, 5.0], [9.0, 7.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "condition": ["A", "A", "B"],
                "cell_type": ["c1", "c1", "c2"],
                "treatment": ["t", "t", "t"],
            }
        ),
    )
    first.var_names = ["g1", "g2"]
    chunk = tmp_path / "chunk.h5ad"
    first.write_h5ad(chunk)
    expected = pd.DataFrame(
        {
            "condition": ["B", "A"],
            "cell_type": ["c2", "c1"],
            "treatment": ["t", "t"],
        }
    )
    output = tmp_path / "centroids.npz"
    summary = MODULE.aggregate_prediction_chunks([chunk], expected, output)
    saved = np.load(output)
    np.testing.assert_allclose(saved["predictions"], [[9.0, 7.0], [2.0, 4.0]])
    assert saved["gene_names"].tolist() == ["g1", "g2"]
    assert saved["gene_names"].dtype.kind == "U"
    assert summary.n_control_predictions.tolist() == [1, 2]
