#!/usr/bin/env python3
"""E229 blocked: incomplete outer-target provenance; gate is testable only."""
import numpy as np
from scipy.stats import rankdata

def quality_gate(magnitude, learned, source_counts, imputed):
    arrays=[np.asarray(v) for v in (magnitude,learned,source_counts,imputed)]
    if not arrays[0].size or any(a.shape!=arrays[0].shape or a.ndim!=1 for a in arrays):
        raise ValueError('Expect nonempty aligned vectors')
    if not all(np.isfinite(a).all() for a in arrays): raise ValueError('Nonfinite gate input')
    m,h,c,i=arrays
    if not np.isin(c,[0,1,2,3]).all() or not np.isin(i,[0,1]).all(): raise ValueError('Invalid support or missingness')
    mr=rankdata(m,method='average')/len(m); hr=rankdata(h,method='average')/len(h)
    return np.where((c==3)&(i==0),hr,mr)

if __name__=='__main__':
    raise SystemExit('E229 BLOCKED_INPUT_PROVENANCE: no valid results; see AUDIT.md. '
                     'E228 source features are not globally outer-target isolated.')
