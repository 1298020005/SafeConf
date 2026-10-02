#!/usr/bin/env python3
"""Check transcribed numeric tables and original-source hashes; no truth arrays."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

P=Path(__file__).resolve().parent
ROOT=P.parent.parent
text=(P/'PAPER.md').read_text()
lines=text.splitlines()
checks=[]

def table_rows(number):
    start=next(i for i,line in enumerate(lines) if line.startswith('| Table '+str(number)+'.'))
    rows=[]
    for line in lines[start+2:]:
        if not line.startswith('|'):break
        rows.append([cell.strip() for cell in line.strip().strip('|').split('|')])
    return rows

def check(name,display,expected):
    want=f'{float(expected):.4f}'
    checks.append({'name':name,'display':display,'expected':float(expected),'pass':display==want})

core=pd.read_csv(P/'evidence/table2_core.csv')
method_names=['Magnitude','Prediction_hgb','Manual_WeightedHistoryDistance','Learned_WeightedHistoryDistance','Manual_hgb','Learned_hgb']
for row,method in zip(table_rows(2),method_names):
    for col,line in enumerate(['Exphormer_to_GAT','GAT_to_Exphormer','TxPert_to_McFaline'],1):
        found=core[(core.line==line)&(core.method==method)]
        assert len(found)==1,(line,method)
        check(f'Table2/{line}/{method}',row[col],found.iloc[0].utility20)
feedback=pd.read_csv(P/'evidence/table4_feedback.csv')
for row in table_rows(5):
    budget=float(row[0].removesuffix('%'))/100
    for col,method in enumerate(['Shared','TargetOnly_HGB','PublicTarget_HGB','SharedTarget_HGB','ResidualHGB'],1):
        found=feedback[(feedback.budget==budget)&(feedback.method==method)]
        assert len(found)==1,(budget,method)
        check(f'Table5/{budget}/{method}',row[col],found.iloc[0].utility20)
methods=pd.read_csv(P/'evidence/frangieh_methods.csv')
pairs=pd.read_csv(P/'evidence/frangieh_pairs.csv')
for row,direction in zip(table_rows(6),['GEARS_to_scGPT','scGPT_to_GEARS']):
    for col,method in enumerate(['PublicHGB','Magnitude'],1):
        found=methods[(methods.direction==direction)&(methods.method==method)&(methods.scope=='heldout_context')&(methods.aggregation=='risk_fold')]
        assert len(found)==1
        check(f'Table6/{direction}/{method}',row[col],found.iloc[0].utility20)
    found=pairs[(pairs.direction==direction)&(pairs.method=='PublicHGB')&(pairs.baseline=='WeightedHistoryDistance')&(pairs.scope=='heldout_context')]
    assert len(found)==1
    r=found.iloc[0]
    check(f'Table6/{direction}/paired_delta',row[3],r.delta_utility20)
    checks.append({'name':f'Table6/{direction}/interval','display':row[4],'pass':row[4]==f'[{r.ci95_lower:.4f}, {r.ci95_upper:.4f}]'})
align=pd.read_csv(P/'evidence/alignment_reference_methods.csv').set_index('method')
rows=table_rows(7)
check('Table7/guide',rows[0][1],align.loc['FrozenGuideEqualReference'].utility20)
check('Table7/cell',rows[1][1],align.loc['CellWeightedReference'].utility20)
alignpairs=pd.read_csv(P/'evidence/alignment_reference_pairs.csv')
for i,baseline in [(1,'FrozenGuideEqualReference'),(2,'NegativeHistorySupport')]:
    r=alignpairs[alignpairs.method_b==baseline].iloc[0]
    check(f'Table7/{baseline}/delta',rows[i][2],r.delta_utility20)
    checks.append({'name':f'Table7/{baseline}/interval','display':rows[i][3],'pass':rows[i][3]==f'[{r.ci95_lower:.4f}, {r.ci95_upper:.4f}]'})
public=pd.read_csv(P/'evidence/publicset_macro_metric_intervals.csv')
public_scopes=['Source/TxPert_GAT','Source/TxPert_Exphormer','McFaline/DecoderOnly']
public_methods=['B0_SupportMean','B1_HGB','B2_Pointwise','B3_DeepSets','Magnitude',
                'NegativeHistorySupport','NearestControl','SameContextSupport','SameConditionSupport']
for row,method in zip(table_rows(8),public_methods):
    for col,scope in enumerate(public_scopes,1):
        found=public[(public.scope==scope)&(public.readout=='risk')&
                     (public.builder==method)&(public.metric=='utility20')]
        assert len(found)==1,(scope,method)
        check(f'Table8/{scope}/{method}',row[col],found.iloc[0].value)
paired=pd.read_csv(P/'evidence/publicset_paired_comparisons.csv')
matched=pd.read_csv(P/'evidence/publicset_strong_simple_comparisons.csv')
comparisons=[(scope,'B2_Pointwise','B1_HGB') for scope in public_scopes]+[
    (scope,'B3_DeepSets','B2_Pointwise') for scope in public_scopes]+[
    ('McFaline/DecoderOnly',candidate,'SameContextSupport') for candidate in ['B2_Pointwise','B3_DeepSets']]
for row,(scope,candidate,comparator) in zip(table_rows(9),comparisons):
    frame=matched if comparator=='SameContextSupport' else paired
    found=frame[(frame.scope==scope)&(frame.readout=='risk')&(frame.metric=='utility20')&
                (frame.candidate==candidate)&(frame.comparator==comparator)]
    assert len(found)==1
    r=found.iloc[0]
    prefix=f'Table9/{scope}/{candidate}/{comparator}'
    check(prefix+'/delta',row[1],r.delta)
    checks.append({'name':prefix+'/interval','display':row[2],
                   'pass':row[2]==f'[{r.ci95_lower:.4f}, {r.ci95_upper:.4f}]'})
    checks.append({'name':prefix+'/nonnegative','display':row[3],
                   'pass':row[3]==f'{100*r.nonnegative_strata_fraction:.1f}%'})
manifest=json.loads((P/'evidence/RESULT_MANIFEST.json').read_text())
hash_checks=[]
for entry in manifest['source_files']:
    source=ROOT/entry['source_relative_to_closure']
    copied=P/entry['paper_artifact']
    expected=entry['sha256']
    ok=hashlib.sha256(source.read_bytes()).hexdigest()==expected==hashlib.sha256(copied.read_bytes()).hexdigest()
    hash_checks.append({'source':entry['source_relative_to_closure'],'pass':ok})
tex=(P/'main.tex').read_text()
environments={s:tex.count(r'\begin{'+s+'}')==tex.count(r'\end{'+s+'}') for s in ['document','abstract','table*','tabular','figure*']}
build=json.loads((P/'build/BUILD_STATUS.json').read_text())
artifacts=[('manuscript_source',P/'PAPER.md','paper_sha256'),
           ('supplement_source',P/'SUPPLEMENT.md','supplement_sha256'),
           ('manuscript_pdf',P/'build/main.pdf','pdf_sha256'),
           ('supplement_pdf',P/'build/supplement.pdf','supplement_pdf_sha256')]
for name,path,key in artifacts:
    checks.append({'name':name+'/fresh_hash','pass':path.is_file() and
                   hashlib.sha256(path.read_bytes()).hexdigest()==build.get(key)})
experimental_markers=re.findall(r'\[\[REGISTERED_[^\]]+\]\]|\bPending result slot\b|\bPublicSet training is in progress\b',
                               text+(P/'SUPPLEMENT.md').read_text())
checks.append({'name':'no_experimental_placeholders','pass':not experimental_markers,
               'remaining':experimental_markers})
checks.append({'name':'both_pdfs_compiled','pass':build.get('compile_status')=='COMPILED'
               and build.get('supplement_compiled') is True})
status={'passed':all(c['pass'] for c in checks+hash_checks) and all(environments.values()),
        'display_cells_checked':len(checks),'manifest_hashes_checked':len(hash_checks),
        'latex_environments_balanced':all(environments.values()),'latex_compilation':build['compile_status'],
        'author_placeholders_intentionally_retained':True,
        'new_model_fits':0,'new_test_truth_reads':0,'checks':checks,'hash_checks':hash_checks}
(P/'build/NUMERIC_CHECKS.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({k:v for k,v in status.items() if k not in ['checks','hash_checks']}))
if not status['passed']:raise SystemExit(1)
