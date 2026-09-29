import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "tools/scripts/supervise_mcfaline_validation.py"
SPEC = importlib.util.spec_from_file_location("mcfaline_supervisor", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_best_checkpoint_requires_one_retained_checkpoint(tmp_path):
    checkpoint = tmp_path / "epoch=2.ckpt"
    checkpoint.write_bytes(b"best")
    assert MOD.best_checkpoint(tmp_path) == checkpoint


def test_process_has_impossible_contract_is_false():
    assert not MOD.process_has("__SafeConf_impossible_process_contract_9d68130c__")
