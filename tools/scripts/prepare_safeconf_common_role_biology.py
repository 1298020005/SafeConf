#!/usr/bin/env python3
"""Common-axis role-specific biology. Existing validation audit definitions.

Training and validation expression alone enter truth/baseline statistics.
The fixed alpha remains 0.25. The same TRAIN cell-mean effect baseline is
used for validation and test prediction conversion; test biology is withheld.
"""
from pathlib import Path
import hashlib,json,sys
import h5py,numpy as np,pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.scripts import run_safeconf_research_closure as closure
class CompetenceFailure(RuntimeError):pass

def categorical(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Group):
        categories = np.asarray(node["categories"]).astype(str)
        codes = np.asarray(node["codes"], dtype=int)
        values = np.empty(len(codes), dtype=object)
        values[codes < 0] = None
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values.astype(str)
    return np.asarray(node).astype(str)

def factorize(frame: pd.DataFrame, columns: list[str], mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    codes = np.full(len(frame), -1, dtype=np.int64)
    selected = frame.loc[mask, columns].astype(str)
    local, uniques = pd.factorize(pd.MultiIndex.from_frame(selected), sort=True)
    codes[np.flatnonzero(mask)] = local
    keys = uniques.to_frame(index=False)
    keys.columns = columns
    return codes, keys

def accumulate(sums: np.ndarray, counts: np.ndarray, codes: np.ndarray, values: np.ndarray) -> None:
    valid = codes >= 0
    if not valid.any():
        return
    use_codes, use_values = codes[valid], values[valid]
    order = np.argsort(use_codes, kind="stable")
    use_codes, use_values = use_codes[order], use_values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(use_codes)) + 1]
    unique = use_codes[starts]
    sums[unique] += np.add.reduceat(use_values, starts, axis=0)
    counts[unique] += np.diff(np.r_[starts, len(use_codes)])

def aggregate_truth_and_baselines(
    h5ad: Path, split_path: Path, gene_names: list[str], chunk_rows: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = pd.read_csv(split_path, header=None, names=["cell_id", "split"])
    split_map = pd.Series(split.split.to_numpy(str), index=split.cell_id.astype(str)).to_dict()
    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        cell_ids = categorical(obs, "_index")
        roles = np.asarray([split_map.get(x, "missing") for x in cell_ids], dtype=object)
        if (roles == "missing").any():
            raise CompetenceFailure("official split failed to align")
        control = categorical(obs, "control") == "1"
        frame = pd.DataFrame({
            "perturbation": categorical(obs, "perturbation"),
            "context": categorical(obs, "cell_type"),
            "treatment": categorical(obs, "treatment"),
        })
        frame["task_id"] = frame.perturbation + "::" + frame.context + "::" + frame.treatment
        masks = {
            "train_task": (roles == "train") & ~control,
            "train_control": (roles == "train") & control,
            "val_task": (roles == "val") & ~control,
            "val_control": (roles == "val") & control,
        }
        specs = []
        for name, mask in masks.items():
            columns = ["context", "treatment"] if "control" in name else ["task_id", "perturbation", "context", "treatment"]
            codes, keys = factorize(frame, columns, mask)
            specs.append((name, codes, keys))
        all_genes = np.asarray(handle["var"]["gene_name"]).astype(str)
        index = pd.Series(np.arange(len(all_genes)), index=all_genes)
        if not set(gene_names) <= set(index.index):
            raise CompetenceFailure("registered gene contract does not align")
        selected = index.loc[gene_names].to_numpy(int)
        if len(selected) != len(gene_names) or len(selected) < 2000:
            raise CompetenceFailure("registered common-axis gene contract failed")
        aggregates = {
            name: [np.zeros((len(keys), len(selected)), dtype=np.float64), np.zeros(len(keys), dtype=np.int64)]
            for name, _, keys in specs
        }
        x = handle["X"]
        indptr = x["indptr"]
        for start in range(0, len(frame), chunk_rows):
            end = min(start + chunk_rows, len(frame))
            p0, p1 = int(indptr[start]), int(indptr[end])
            block = sparse.csr_matrix((
                np.asarray(x["data"][p0:p1]),
                np.asarray(x["indices"][p0:p1]),
                np.asarray(indptr[start:end + 1], dtype=np.int64) - p0,
            ), shape=(end - start, len(all_genes)))[:, selected].toarray()
            for name, codes, _ in specs:
                accumulate(aggregates[name][0], aggregates[name][1], codes[start:end], block)
            if start % 100000 == 0:
                print(f"[CompetenceTruth] rows {start}:{end}/{len(frame)}", flush=True)
    means = {}
    keys_by_name = {}
    for name, _, keys in specs:
        sums, counts = aggregates[name]
        if (counts == 0).any():
            raise CompetenceFailure(f"empty aggregate: {name}")
        means[name] = sums / counts[:, None]
        keys_by_name[name] = keys
    train_controls = {
        tuple(row): means["train_control"][i]
        for i, row in enumerate(keys_by_name["train_control"].itertuples(index=False, name=None))
    }
    val_controls = {
        tuple(row): means["val_control"][i]
        for i, row in enumerate(keys_by_name["val_control"].itertuples(index=False, name=None))
    }
    train_effects_by_state: dict[tuple[str, str], list[np.ndarray]] = {}
    for i, row in enumerate(keys_by_name["train_task"].itertuples(index=False)):
        state = (str(row.context), str(row.treatment))
        if state in train_controls:
            train_effects_by_state.setdefault(state, []).append(means["train_task"][i] - train_controls[state])
    state_mean = {state: np.mean(vectors, axis=0) for state, vectors in train_effects_by_state.items()}
    metadata, truth, state_baseline, validation_controls = [], [], [], []
    for i, row in enumerate(keys_by_name["val_task"].itertuples(index=False)):
        state = (str(row.context), str(row.treatment))
        if state not in val_controls or state not in state_mean:
            continue
        metadata.append({
            "task_id": str(row.task_id), "perturbation": str(row.perturbation),
            "context": state[0], "treatment": state[1], "stratum": f"{state[0]}::{state[1]}",
        })
        truth.append(means["val_task"][i] - val_controls[state])
        state_baseline.append(state_mean[state])
        validation_controls.append(val_controls[state])
    truth = np.asarray(truth)
    return pd.DataFrame(metadata), truth, np.zeros_like(truth), np.asarray(state_baseline), np.asarray(validation_controls)

def main():
    out=closure.RUNTIME/'common_gene_axis'
    genes=json.loads((out/'GENE_IDS.json').read_text())
    h5ad=Path('/home/yyf/data/perturbench_mcfaline23_official/mcfaline23_gxe_processed.h5ad')
    split=h5ad.parent/'splits/mcfaline23_gxe_splits/full_covariate_split.csv'
    meta,truth,zero,baseline,controls=aggregate_truth_and_baselines(h5ad,split,genes,2000)
    predictions=pd.read_csv(out/'VALIDATION_TASKS.csv')
    order=pd.Series(np.arange(len(meta)),index=meta.task_id).loc[predictions.task_id].to_numpy(int)
    meta=meta.iloc[order].reset_index(drop=True);truth=truth[order];baseline=baseline[order];controls=controls[order]
    pred=np.load(out/'VALIDATION_PREDICTED_STATES.npy')-controls
    calibrated=.25*pred+.75*baseline
    for name,array in [('VALIDATION_TRUE_EFFECTS',truth),('VALIDATION_CONTROLS',controls),
                       ('VALIDATION_RAW_EFFECTS',pred),('VALIDATION_TRAIN_STATE_MEAN_EFFECTS',baseline),
                       ('VALIDATION_CALIBRATED_EFFECTS',calibrated)]:
        closure.tx.atomic_npy(out/(name+'.npy'),array.astype(np.float32))
    closure.tx.atomic_csv(out/'VALIDATION_BIOLOGY_TASKS.csv',meta)
    states={}
    for (context,treatment),group in meta.groupby(['context','treatment']):
        array=baseline[group.index.to_numpy()]
        if np.max(np.abs(array-array[0]))>1e-10:raise RuntimeError('TRAIN state baseline differs within state')
        states[context+'::'+treatment]=array[0]
    np.savez(out/'TRAIN_STATE_MEAN_EFFECTS.npz',**states)
    closure.tx.atomic_json(closure.OUT/'common_gene_axis/ROLE_BIOLOGY_STATUS.json',{
        'status':'COMPLETE','n_validation_tasks':len(meta),'n_genes':len(genes),'alpha':.25,
        'state_baseline':'unweighted average of cell-weighted TRAIN task effects; TRAIN controls',
        'model_effect':'frozen generated treated mean minus observed same-partition control',
        'test_treated_expression_used':False,'role':'SEEN_POST_CONFIRMATION',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print('ROLE_BIOLOGY_COMPLETE',len(meta),len(genes),flush=True)
if __name__=='__main__':main()
