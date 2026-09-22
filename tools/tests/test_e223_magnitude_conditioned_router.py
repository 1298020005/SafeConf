from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


MODULE = Path(__file__).parents[1] / 'scripts' / 'run_e223_magnitude_conditioned_router.py'
spec = importlib.util.spec_from_file_location('e223', MODULE)
e223 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e223)


def test_rules_include_magnitude_and_scenario_maps():
    names = [x['name'] for x in e223.rules()]
    assert names[0] == 'magnitude'
    assert 'scenario_a' in names
    assert 'anchored_evidence' in names


def test_one_sided_keeps_magnitude_anchor():
    frame = pd.DataFrame({'m': [0.1, 0.2], 's': [0.9, 0.1], 'd': [0.1, 0.1],
                          'cn': [0.1, 0.1], 'pn': [0.1, 0.1], 'ss': [0.1, 0.1],
                          'setting': ['context_unseen_row', 'context_unseen_row']})
    rule = next(x for x in e223.rules() if x['name'] == 'one_sided_s0.50')
    values = e223.score(frame, rule)
    np.testing.assert_allclose(values, [0.5, 0.2])


def test_metrics_is_finite():
    out = e223.metrics(np.array([.1, .2, .3, .4]), np.array([.1, .3, .2, .5]))
    assert np.isfinite(out['spearman'])
    assert np.isfinite(out['utility_20'])
