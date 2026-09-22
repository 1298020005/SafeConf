from pathlib import Path
import importlib.util

MODULE = Path(__file__).parents[1] / 'scripts' / 'run_e224_scenario_gate_router.py'
spec = importlib.util.spec_from_file_location('e224', MODULE)
e224 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e224)


def test_hard_gate_has_zero_context_weight():
    rule = next(x for x in e224.rules() if x['name'] == 'gate_hard_context_only')
    assert rule['params']['context_unseen_row'] == 0.0
    assert rule['params']['random_missing_pair'] == 1.0
