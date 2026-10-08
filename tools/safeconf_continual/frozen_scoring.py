"""One label-free scoring entrypoint for rules and frozen supervised outputs."""
from __future__ import annotations
import numpy as np


def score_task(task, frozen_config):
    """Return score, evidence state, version; never accesses query truth.

CDF arrays are fitted before this call on authorized preparation features.
For a supervised candidate, ``frozen_model_score`` is produced by the frozen
model, with its training provenance bound in ``frozen_config``.
"""
    method=frozen_config['method'];version=frozen_config['version']
    available=bool(task.get('public_available',False))
    state='LEGAL_HISTORY' if available else 'NO_HISTORY_AMPLITUDE_FALLBACK'
    def transform(channel,value):
        values=np.asarray(frozen_config['channel_cdfs'][channel],float)
        if len(values)<2 or not np.isfinite(values).all() or np.any(np.diff(values)<0):
            raise ValueError('finite sorted training-side channel CDF required')
        if not np.isfinite(value):raise ValueError('finite rule input required')
        return float((np.searchsorted(values,value,'left')+np.searchsorted(values,value,'right'))/(2*len(values)))
    if method=='FrozenDirectRule':
        value=float(task['frozen_rule_score']);state=str(task.get('evidence_status','FROZEN_DIRECT_RULE'))
    elif method=='FrozenSupervised':
        value=float(task['frozen_model_score']);state=str(task.get('evidence_status','FROZEN_SUPERVISED'))
    elif method=='Magnitude' or not available and method in ['PublicRule','HistorySupport']:
        value=transform('Amplitude',task['predicted_magnitude'])
    elif method=='PublicRule':value=transform('Public',task['public_raw'])
    elif method=='HistorySupport':value=transform('Support',task['support_raw'])
    elif method=='Similarity':
        similarity=float(task['similarity_raw'])
        if np.isfinite(similarity):value=transform('Similarity',similarity)
        else:
            value=transform('Amplitude',task['predicted_magnitude']);state='NO_EMBEDDING_AMPLITUDE_FALLBACK'
    else:raise ValueError('unregistered frozen scoring method')
    if not np.isfinite(value):raise ValueError('nonfinite final risk score')
    return value,state,version
