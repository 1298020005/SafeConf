#!/usr/bin/env python3
"""Verify the report package, released numerical tables and figure manifests.

This reads released summaries and files created by this report task. It does
not fit models, read expression arrays or access sealed evaluation truth.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/组会汇报/20261008_最终报告'
EVID = OUT / 'evidence'


def read(name):
    return json.loads((EVID / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    documents = sorted(OUT.glob('*.md'))
    assert len(documents) == 2
    for document in documents:
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            if target.startswith(('https://', 'http://')):
                continue
            assert (document.parent / target).exists(), target
    layout = read('LAYOUT_CHECK.json')
    assert len(layout) == 8
    assert all(not item['text_overlaps'] and not item['clipped_labels'] for item in layout)
    manifest = read('FIGURE_MANIFEST.json')
    for name, digest in manifest['figures'].items():
        assert sha(OUT / 'figures' / name) == digest, name
    for name, digest in manifest['original_hashes_unchanged'].items():
        assert sha(OUT / 'figures' / 'originals' / name) == digest
    assert manifest['outlined_svg_exported']
    coverage = read('TERM_COVERAGE_CHECK.json')
    assert not coverage['unexplained_english_tokens']
    assert coverage['registry_entries'] == 213
    sources = rows(EVID / 'RESULT_SOURCE_BINDINGS.csv')
    for item in sources:
        snapshot = OUT / item['review_snapshot']
        assert sha(snapshot) == item['snapshot_sha256']
        if Path(item['path']).suffix == '.md':
            value = json.loads(snapshot.read_text())
            assert hashlib.sha256(value['source_text'].encode()).hexdigest() == item['sha256']
        else:
            assert sha(snapshot) == item['sha256']
    review = {item['method']: item for item in rows(EVID / 'canonical_sources/global_review.csv')}
    public = review['PublicRule']
    magnitude = review['Amplitude']
    assert int(float(public['high_error_found'])) == 22
    assert int(float(magnitude['high_error_found'])) == 4
    # These are the released global review endpoints; context-macro U20 is
    # verified separately and is never substituted for remaining error.
    remaining_key = next(key for key in public if 'remaining' in key and 'mean' in key)
    public_mean = float(public[remaining_key])
    magnitude_mean = float(magnitude[remaining_key])
    reduction = (magnitude_mean - public_mean) / magnitude_mean
    assert math.isclose(reduction, .07070713149896503, abs_tol=1e-7)
    content = rows(EVID / 'current_public_content/RESULTS.csv')
    public_u20 = float(next(item['u20'] for item in content if item['method'] == 'PublicRule'))
    controls = [item for item in content if item['method'].startswith('CurrentContentNull')]
    assert len(controls) == 5
    assert math.isclose(public_u20, .8376675521178553, abs_tol=1e-12)
    for item in controls:
        assert math.isclose(public_u20 - float(item['u20']), float(item['delta_utility20']), abs_tol=1e-12)
        assert int(float(item['bootstrap_replicates'])) == 5000
        assert int(float(item['n_tasks'])) == 212
        assert int(float(item['n_clusters'])) == 152
    assert sum(float(item['ci95_lower']) > 0 for item in controls) == 4
    assert all(float(item['delta_utility20']) > 0 for item in controls)
    permutation = rows(EVID / 'current_public_content/PERMUTATION_AUDIT.csv')
    assert len(permutation) == 5
    assert all(float(item['moved_fraction']) == 1 and int(item['forbidden_donors_used']) == 0 for item in permutation)
    protected = subprocess.check_output(['ps', '-p', '1873824,2428149', '-o', 'pid='], text=True).split()
    assert set(protected) == {'1873824', '2428149'}
    scripts = [ROOT / 'tools/scripts' / name for name in [
        'build_safeconf_final_report_figures_v1.py', 'build_safeconf_plain_glossary_v1.py',
        'assemble_safeconf_report_evidence_v1.py', 'verify_safeconf_final_report_v1.py',
        'run_safeconf_native_similarity_dev_diagnostic_v1.py', 'run_safeconf_current_public_content_audit_v1.py']]
    receipt = {
        'status': 'REPORT_PACKAGE_READY',
        'scope': 'research evolution, publication decision, two Markdown documents and separate figures',
        'utc_verified': datetime.now(timezone.utc).isoformat(),
        'mainline': 'experimental-reference risk auditing; TCBB submission direction',
        'current_default': 'PublicRule on the registered McFaline contract',
        'source_role': 'compatible-predictor shared learning with explicit strong-baseline comparisons',
        'target_role': 'candidate updates selected on allowed development data',
        'papers_and_submission_materials_generated_this_task': False,
        'journal_acceptance_assessed_or_guaranteed': False,
        'new_gpu_training_hours': 0, 'new_download_bytes': 0,
        'new_cpu_diagnostic_fits': 36, 'new_public_content_fits': 0,
        'sealed_test_truth_opened_this_task': False,
        'already_seen_target_errors_used_for_content_diagnostic': True,
        'protected_processes_present': protected,
        'history_inventory': read('HISTORY_COVERAGE.json'),
        'term_coverage': coverage,
        'figure_count': len(layout), 'text_overlaps': 0, 'clipped_text': 0,
        'visual_inspection': 'all eight PNG figures inspected; corrected formula arrows, title and annotation placement',
        'original_three_plots_and_composite_hashes_unchanged': True,
        'global_review': {'n_tasks': 212, 'k': 43, 'public_hits': 22, 'magnitude_hits': 4,
            'extra_hits': 18, 'remaining_error_reduction_vs_magnitude': reduction},
        'current_content': {'public_u20': public_u20, 'fixed_permutations': 5,
            'positive_point_deltas': 5, 'positive_ci_lower_bounds': 4, 'remaining_ci_crosses_zero': True},
        'document_sha256': {path.name: sha(path) for path in documents},
        'script_sha256': {str(path.relative_to(ROOT)): sha(path) for path in scripts},
        'next_research_action': 'measurement-endpoint diagnostic, then frozen Norman/GWPS cross-study replication',
        'reproduce_figures': f'/home/miniconda/bin/python {ROOT}/tools/scripts/build_safeconf_final_report_figures_v1.py',
        'rebuild_glossary': f'/home/miniconda/bin/python {ROOT}/tools/scripts/build_safeconf_plain_glossary_v1.py',
        'assemble_evidence': f'/home/miniconda/bin/python {ROOT}/tools/scripts/assemble_safeconf_report_evidence_v1.py',
        'verify_package': f'/home/miniconda/bin/python {Path(__file__).resolve()}'
    }
    (EVID / 'COMPLETION_AUDIT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'figures': len(layout), 'terms': coverage['registry_entries'],
        'public_extra_hits': 18, 'new_gpu_hours': 0, 'protected_processes_present': protected}, ensure_ascii=False))


if __name__ == '__main__':
    main()
