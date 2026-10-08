#!/usr/bin/env python3
"""Index historical reports and assemble the final report's source receipts.

The inventory reads narrative reports/status metadata, never expression or
sealed truth arrays. Status words are indexed as historical text, not gates.
"""
from pathlib import Path
import csv,hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/组会汇报/20261008_最终报告'
EVID=OUT/'evidence'
STAGE=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001'
GOAL=Path('/home/yyf/runtime_artifacts/safeconf_goal_report_20261008_v1')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory():
    result=[]
    for origin,parent in [('repository',ROOT/'docs/实验结果'),('old_runtime',Path('/home/yyf/safeconf_runtime/outputs'))]:
        for directory in sorted(parent.iterdir()):
            if not directory.is_dir():continue
            docs=[]
            for f in directory.rglob('*'):
                if not f.is_file() or 'paper' in f.relative_to(directory).parts:continue
                if f.suffix=='.md' and any(x in f.name.lower() for x in ['report','decision','interpret','readme','报告','结论','说明','看懂']):docs.append(f)
                elif f.suffix=='.json' and any(x in f.name.lower() for x in ['run_status','execution_status','final_decision','component_decision']):docs.append(f)
            date=re.search(r'(202[0-9][01][0-9][0-3][0-9])',directory.name)
            markers=set();heads=[];digests=[]
            for f in docs:
                raw=f.read_bytes();text=raw.decode('utf-8',errors='replace')
                markers.update(re.findall(r'\b(?:PASS|FAIL|SEEN|ABSTAIN|COMPLETE|NO_TARGET_REPLICATION)\b',text))
                line=next((x.strip('# ') for x in text.splitlines() if x.strip()),'')
                if len(heads)<3:heads.append(line[:150])
                digests.append(f'{f.relative_to(directory)} {hashlib.sha256(raw).hexdigest()}')
            result.append({'origin':origin,'experiment_directory':str(directory),'declared_date':date.group(1) if date else '',
                'metadata_reports_read':len(docs),'reported_status_markers':'|'.join(sorted(markers)),
                'headings':' / '.join(heads),'report_manifest_sha256':hashlib.sha256('\n'.join(digests).encode()).hexdigest(),
                'role_note':'historical inventory; markers do not establish final scientific status'})
    with (EVID/'EXPERIMENT_INVENTORY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator='\n');w.writeheader();w.writerows(result)
    return result


def main():
    EVID.mkdir(parents=True,exist_ok=True)
    inv=inventory()
    named=[
      ('May history',Path('/home/yyf/safeconf_runtime/outputs/current_project_explanation_20260530/一份看懂全部.md')),
      ('June corrected main',ROOT/'docs/实验结果/Formal_main_20260604/corrected_v3_drop_blank_1000_20260609/reports/CORRECTED_7MAIN_DECISION.md'),
      ('June fold-safe correction',Path('/home/yyf/safeconf_runtime/outputs/e1_e4_preregistered_formal_foldsafe_20260614/E1_E4_GATE_REPORT.md')),
      ('E90',ROOT/'docs/实验结果/E90_gene_hard_setting_matrix_20260712/reports/E90_REPORT.md'),
      ('E153',ROOT/'docs/实验结果/E153_eight_study_formal_meta_20260714/reports/E153_REPORT.md'),
      ('E172',ROOT/'docs/实验结果/E172_primary_cd4_fresh_targets_20260718/postgate_release/reports/E172_JOINT_POSTGATE_REPORT.md'),
      ('E181',ROOT/'docs/实验结果/E181_registered_family_hilbert_certificate_20260724/reports/E181_REPORT.md'),
      ('E189',ROOT/'docs/实验结果/E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md'),
      ('E201',ROOT/'docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/reports/E201_CORE_REPORT.md'),
      ('SafeConf-M',ROOT/'docs/组会汇报_20260913.md'),
      ('Current algorithms',STAGE/'publicset_execution_v1/FINAL_MODEL_SETTINGS.json'),
      ('Current decisions',STAGE/'publicset_execution_v1/FINAL_EXPERIMENTAL_DECISION.md'),
      ('Source P-only comparison',STAGE/'common_gene_axis/results/PAIRED_CLUSTER_BOOTSTRAP.csv'),
      ('Source repaired metrics',STAGE/'data_model_feedback_20261003_v1/review_repair_20261007_v1/SOURCE_REAL_COMPARISON.csv'),
      ('Target Public budgets',STAGE/'data_model_feedback_20261003_v1/review_repair_20261007_v1/NATIVE_PUBLIC_BUDGET_TABLE.csv'),
      ('Global review',STAGE/'data_model_feedback_20261003_v1/review_repair_20261007_v1/GLOBAL_20_PERCENT_REVIEW.csv'),
      ('Source magnitude',ROOT/'docs/组会汇报/20261008_科学示意_v2/SOURCE_MAGNITUDE_PAIRED_BOOTSTRAP.csv'),
      ('SAMS competence',STAGE/'data_model_feedback_20261003_v1/sams/VALIDATION_COMPETENCE.json'),
      ('SAMS transfer',STAGE/'data_model_feedback_20261003_v1/system_evidence_v042/SAMS_CROSSFAMILY_MACRO.csv'),
      ('Native diagnostic',GOAL/'native_similarity_dev/SUMMARY.csv'),
      ('Current content',GOAL/'current_public_content/RESULTS.csv'),
    ]
    sources=[]
    snapshots=EVID/'canonical_sources';snapshots.mkdir(exist_ok=True)
    for topic,p in named:
        if not p.is_file():raise FileNotFoundError(p)
        raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        stem=re.sub(r'[^a-z0-9]+','_',topic.lower()).strip('_')
        if p.suffix.lower()=='.md':
            snapshot=snapshots/(stem+'.json')
            snapshot.write_text(json.dumps({'source_path':str(p),'source_sha256':digest,
                'role':'read-only historical report snapshot, not a new manuscript',
                'source_text':raw.decode('utf-8')},ensure_ascii=False,indent=2)+'\n')
        else:
            snapshot=snapshots/(stem+p.suffix.lower());snapshot.write_bytes(raw)
        sources.append({'topic':topic,'path':str(p),'sha256':digest,'bytes':len(raw),
            'review_snapshot':str(snapshot.relative_to(OUT)),'snapshot_sha256':sha(snapshot)})
    with (EVID/'RESULT_SOURCE_BINDINGS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sources[0]),lineterminator='\n');w.writeheader();w.writerows(sources)
    for sub in ['native_similarity_dev','current_public_content','current_evidence']:
        dest=EVID/sub;dest.mkdir(exist_ok=True)
        for p in (GOAL/sub).iterdir():
            if p.suffix in ['.csv','.json','.py']:
                shutil.copyfile(p,dest/p.name)
    web=[
      ('TCBB rank','CCF B category','https://www.ccf.org.cn/c/2019-05-14/665229.shtml','official indexed page'),
      ('TCBB scope','methods/statistics/programs/databases','https://www.computer.org/digital-library/journals/bb/cfp-computational-biology-bioinformatics','official indexed scope'),
      ('PertEMA','own-OOF-error post-hoc estimation','https://github.com/OfficialBishal/PertEMA','official README plus local fixed source'),
      ('Risk Advisor','existing black-box failure meta-learner','https://link.springer.com/article/10.1007/s10994-022-06248-y','primary paper abstract'),
      ('MORPH','active-learning error function and next experiments','https://pmc.ncbi.nlm.nih.gov/articles/PMC12236822/','author paper and official repository'),
      ('MORPH code','author implementation','https://github.com/uhlerlab/MORPH','official repository'),
      ('PRESCRIBE','joint model/data uncertainty','https://papers.nips.cc/paper_files/paper/2025/hash/d6383e7643415842b48a5077a1b09c98-Abstract-Conference.html','conference primary source'),
      ('PerturbMap','paired cross-context response mapping','https://arxiv.org/abs/2607.28090','author preprint'),
      ('Metrics 2026','Deep learning perturbation models can outperform baselines on calibrated metrics','https://www.nature.com/articles/s41587-026-03307-w','publisher title/DOI; published 2026-10-01'),
    ]
    with (EVID/'LITERATURE_SOURCES.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['work','verified_point','primary_url','source_scope']);w.writerows(web)
    labels=json.loads((EVID/'FIGURE_LABELS.json').read_text())
    glossary=(OUT/'SafeConf_名词与图解.md').read_text().lower()
    maintext=(OUT/'SafeConf_研究主线与汇报.md').read_text()
    maintext=re.sub(r'https?://[^\s)]+','',maintext)
    maintext=re.sub(r'\]\([^)]*\)',']',maintext)
    tokens=set()
    for s in [maintext]+[x['label'] for x in labels]:
        s=re.sub(r'\\\[.*?\\\]','',s,flags=re.S)
        s=re.sub(r'\$[^$]*\$','',s)
        s=re.sub(r'\\[A-Za-z]+','',s)
        tokens.update(re.findall(r'[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*',s.lower()))
    missing=[x for x in sorted(tokens) if x not in glossary and x.rstrip('s') not in glossary]
    coverage={'registry_entries':len(json.loads((EVID/'TERM_REGISTRY.json').read_text())),
        'english_tokens_checked':len(tokens),'unexplained_english_tokens':missing,
        'figure_labels_checked':len(labels),'scope':'main report and final figure text; each glossary entry manually describes project role'}
    (EVID/'TERM_COVERAGE_CHECK.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
    if missing:raise RuntimeError(f'unexplained terms: {missing}')
    summary={'indexed_experiment_directories':len(inv),'repository_directories':sum(r['origin']=='repository' for r in inv),
        'old_runtime_directories':sum(r['origin']=='old_runtime' for r in inv),'historical_report_status_files_read':sum(r['metadata_reports_read'] for r in inv),
        'curated_numeric_sources':len(sources),'raw_arrays_recomputed_all':False,
        'scope':'all indexed report/status histories plus selected numerical/contract audits; no sealed truth read'}
    (EVID/'HISTORY_COVERAGE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False));print(json.dumps(coverage,ensure_ascii=False))


if __name__=='__main__':main()
