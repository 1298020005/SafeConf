"""Meaningful input-isolation checks for the registered SAMS driver."""
import ast
import importlib.util
import inspect
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from anndata.io import sparse_dataset, write_elem

PATH = Path(__file__).with_name("run_safeconf_sams_crossfamily_v1.py")
spec = importlib.util.spec_from_file_location("sams_driver", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SamsInputContract(unittest.TestCase):
    def test_guarded_x_only_blocks_test_before_read(self):
        # Compile the actual reader class from the training function, without
        # loading the 8e5-cell data just to test the access guard.
        tree = ast.parse(inspect.getsource(m.training_data))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "GuardedX")
        with tempfile.TemporaryDirectory(dir=m.ROOT) as tmp:
            path = Path(tmp) / "small.h5"
            with h5py.File(path, "w") as h:
                write_elem(h, "X", csr_matrix(np.arange(24).reshape(8, 3)))
                h.create_dataset("layers/counts", data=np.full((8, 3), 999))
            with h5py.File(path, "r") as h:
                reads = []
                class XOnlyFile:
                    def __getitem__(self, key):
                        if key != "X":
                            raise PermissionError("counts/layers forbidden")
                        reads.append(key)
                        return h[key]
                proxy = XOnlyFile()
                ns = {"np": np, "obs": list(range(8)), "allowed": {0, 1, 4},
                      "handle": proxy, "sparse_dataset": sparse_dataset}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(PATH), "exec"), ns)
                x = ns["GuardedX"]()
                np.testing.assert_array_equal(x[[0, 4]].toarray(), [[0, 1, 2], [12, 13, 14]])
                before = len(reads)
                with self.assertRaises(PermissionError):
                    x[[1, 2]]
                self.assertEqual(before, len(reads))
                with self.assertRaises(PermissionError):
                    proxy["layers/counts"]
                self.assertEqual(reads, ["X"])

    def test_no_eager_anndata_or_reference_evaluation_calls(self):
        tree = ast.parse(inspect.getsource(m.training_data))
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn("read_h5ad", calls)
        tree = ast.parse(inspect.getsource(m.train))
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn("test", calls)

    def test_prediction_has_no_data_or_truth_argument(self):
        self.assertEqual(list(inspect.signature(m.predict_metadata).parameters),
                         ["model", "task", "n", "seed", "chunk"])

    def test_data_order_resume_has_no_repeat_or_omission(self):
        sampler=m.ReplayableRandomSampler(73)
        sampler.set_epoch(5)
        full=list(sampler)
        sampler.completed=19
        recovered=m.ReplayableRandomSampler(73)
        recovered.load_state_dict(sampler.state_dict())
        self.assertEqual(list(recovered),full[19:])
        self.assertEqual(sorted(full),list(range(73)))
        recovered.set_epoch(6)
        self.assertEqual(recovered.completed,0)
        self.assertNotEqual(list(recovered),full)

    def test_training_checkpoint_reload_preserves_generation(self):
        import lightning as L
        import torch
        from torch.utils.data import DataLoader
        from lightning.pytorch.loggers import CSVLogger
        model = m.preflight()
        model.train()
        batch = m.generation_batch(model, "GENE_A", "CELL_A", "VEHICLE", 8)
        batch = batch._replace(gene_expression=torch.rand(8,12))
        class Logger(CSVLogger):
            def log_hyperparams(self, params):
                return
        with tempfile.TemporaryDirectory(dir=m.ROOT) as tmp:
            trainer=L.Trainer(accelerator="cpu",devices=1,max_epochs=1,min_epochs=1,max_steps=2,
                logger=Logger(tmp,name="smoke"),enable_checkpointing=False,enable_progress_bar=False,
                enable_model_summary=False,num_sanity_val_steps=0,log_every_n_steps=1)
            loader=DataLoader([batch,batch],batch_size=None)
            trainer.fit(model,train_dataloaders=loader,val_dataloaders=loader)
            path=Path(tmp)/"resumable.ckpt"
            trainer.save_checkpoint(path)
            restored=type(model).load_from_checkpoint(path,weights_only=False,map_location="cpu")
            task={"perturbation":"GENE_A","context":"CELL_A","treatment":"VEHICLE"}
            before,_=m.predict_metadata(model,task,n=8,chunk=8)
            after,_=m.predict_metadata(restored,task,n=8,chunk=8)
            np.testing.assert_allclose(before,after,rtol=1e-5,atol=1e-7)
            payload=torch.load(path,weights_only=False,map_location="cpu")
            self.assertEqual(payload["global_step"],2)
            self.assertTrue(payload["optimizer_states"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SamsInputContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    m.receipt("READER_ISOLATION_TESTS.json", {"status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "tested_driver_sha256": m.sha(PATH), "test_data": "synthetic; no biological expression used"})
    raise SystemExit(0 if result.wasSuccessful() else 1)
