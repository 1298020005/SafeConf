#!/usr/bin/env python3
"""Portable, standard-library validation of the current SafeConf certificate release.

This script does not train models or read raw expression matrices.  It validates
the committed E181/E182/E183 release artifacts, recomputes the headline counts,
checks deterministic inequalities and target-cluster coverage, and emits a small
machine-readable audit bundle.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


TOL = 1e-10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="runtime/current_release_audit",
        help="Output directory (default: runtime/current_release_audit).",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("Unrecognized boolean value: {!r}".format(value))


def close(actual, expected, tolerance=1e-12):
    return abs(float(actual) - float(expected)) <= tolerance


def beta_binomial_cdf(k_max, n, alpha, beta):
    log_beta_ab = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    total = 0.0
    for k in range(k_max + 1):
        log_choose = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
        )
        log_beta_post = (
            math.lgamma(k + alpha)
            + math.lgamma(n - k + beta)
            - math.lgamma(n + alpha + beta)
        )
        total += math.exp(log_choose + log_beta_post - log_beta_ab)
    return total


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail):
        self.checks.append(
            {"check": name, "passed": bool(passed), "detail": str(detail)}
        )

    @property
    def passed(self):
        return all(item["passed"] for item in self.checks)


def verify_hash_table(repo, path, audit, prefix):
    rows = read_csv(path)
    for row in rows:
        declared = Path(row["path"])
        candidate = declared if declared.is_absolute() else repo / declared
        exists = candidate.is_file()
        audit.check(
            "{}_exists_{}".format(prefix, declared.name),
            exists,
            row["path"],
        )
        if not exists:
            continue
        actual_bytes = candidate.stat().st_size
        actual_hash = sha256_file(candidate)
        audit.check(
            "{}_bytes_{}".format(prefix, declared.name),
            actual_bytes == int(row["bytes"]),
            "{} expected={} actual={}".format(
                row["path"], row["bytes"], actual_bytes
            ),
        )
        audit.check(
            "{}_sha256_{}".format(prefix, declared.name),
            actual_hash == row["sha256"],
            row["path"],
        )
    return rows


def verify_e181_manifest(e181_dir, audit):
    manifest = e181_dir / "MANIFEST.sha256"
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = e181_dir / relative
        entries.append(relative)
        audit.check(
            "e181_manifest_{}".format(relative.replace("/", "_")),
            path.is_file() and sha256_file(path) == digest,
            relative,
        )
    audit.check("e181_manifest_nonempty", len(entries) >= 10, len(entries))
    return len(entries)


def recompute_release(repo, audit):
    e181_dir = (
        repo
        / "docs/实验结果/E181_registered_family_hilbert_certificate_20260724"
    )
    e182_dir = (
        repo
        / "docs/实验结果/E182_gse225807_registered_family_20260724"
        / "final_evaluation"
    )
    e183_dir = (
        repo
        / "docs/实验结果/E183_all_study_family_synthesis_20260724"
    )

    e181_manifest_entries = verify_e181_manifest(e181_dir, audit)
    e183_inputs = verify_hash_table(
        repo, e183_dir / "tables/INPUT_HASHES.csv", audit, "e183_input"
    )

    tasks = read_csv(e183_dir / "tables/E183_COMBINED_TASK_CERTIFICATES.csv")
    targets_reported = read_csv(e183_dir / "tables/E183_TARGET_CERTIFICATES.csv")
    studies_reported = read_csv(e183_dir / "tables/E183_STUDY_SUMMARY.csv")
    run_status = read_json(e183_dir / "RUN_STATUS.json")
    e182_status = read_json(e182_dir / "E182_FINAL_SUMMARY.json")

    audit.check("task_count_2433", len(tasks) == 2433, len(tasks))
    audit.check(
        "all_tasks_frozen_10_family",
        all(row["family"] == "frozen_10_seed_family" for row in tasks),
        sorted(set(row["family"] for row in tasks)),
    )
    audit.check(
        "all_families_have_10_members",
        all(int(row["n_members"]) == 10 for row in tasks),
        sorted(set(row["n_members"] for row in tasks)),
    )

    family_lower_violations = 0
    worst_lower_violations = 0
    family_upper_tasks_covered = 0
    worst_upper_tasks_covered = 0
    max_identity_residual = 0.0
    grouped = defaultdict(list)

    for row in tasks:
        family_error = float(row["family_rms_error"])
        family_lower = float(row["diversity_lower"])
        worst_error = float(row["worst_member_error"])
        worst_lower = float(row["diameter_lower"])
        identity = float(row["family_identity_abs_residual"])
        family_upper = float(row["family_upper"])
        worst_upper = float(row["worst_upper"])

        direct_family_violation = family_lower > family_error + TOL
        direct_worst_violation = worst_lower > worst_error + TOL
        family_flag = as_bool(row["family_lower_violation"])
        worst_flag = as_bool(row["worst_lower_violation"])
        family_covered = family_error <= family_upper + TOL
        worst_covered = worst_error <= worst_upper + TOL

        audit.check(
            "family_flag_matches_{}".format(row["task_id"]),
            direct_family_violation == family_flag,
            row["task_id"],
        )
        audit.check(
            "worst_flag_matches_{}".format(row["task_id"]),
            direct_worst_violation == worst_flag,
            row["task_id"],
        )
        audit.check(
            "family_coverage_flag_matches_{}".format(row["task_id"]),
            family_covered == as_bool(row["family_upper_covered"]),
            row["task_id"],
        )
        audit.check(
            "worst_coverage_flag_matches_{}".format(row["task_id"]),
            worst_covered == as_bool(row["worst_upper_covered"]),
            row["task_id"],
        )

        family_lower_violations += int(direct_family_violation)
        worst_lower_violations += int(direct_worst_violation)
        family_upper_tasks_covered += int(family_covered)
        worst_upper_tasks_covered += int(worst_covered)
        max_identity_residual = max(max_identity_residual, identity)
        grouped[(row["study"], row["target_cluster"])].append(
            (family_covered, worst_covered)
        )

    target_rows = {}
    for key, values in grouped.items():
        target_rows[key] = {
            "n_tasks": len(values),
            "family_covered": all(item[0] for item in values),
            "worst_covered": all(item[1] for item in values),
        }

    audit.check("target_cluster_count_737", len(target_rows) == 737, len(target_rows))

    reported_map = {
        (row["study"], row["target_cluster"]): row for row in targets_reported
    }
    audit.check(
        "target_key_set_matches",
        set(target_rows) == set(reported_map),
        "computed={} reported={}".format(len(target_rows), len(reported_map)),
    )
    for key, computed in target_rows.items():
        reported = reported_map[key]
        audit.check(
            "target_task_count_{}_{}".format(key[0], key[1]),
            computed["n_tasks"] == int(reported["n_tasks"]),
            key,
        )
        audit.check(
            "target_family_flag_{}_{}".format(key[0], key[1]),
            computed["family_covered"]
            == as_bool(reported["family_upper_simultaneous_covered"]),
            key,
        )
        audit.check(
            "target_worst_flag_{}_{}".format(key[0], key[1]),
            computed["worst_covered"]
            == as_bool(reported["worst_upper_simultaneous_covered"]),
            key,
        )

    family_upper_targets_covered = sum(
        int(row["family_covered"]) for row in target_rows.values()
    )
    worst_upper_targets_covered = sum(
        int(row["worst_covered"]) for row in target_rows.values()
    )
    studies = sorted(set(row["study"] for row in tasks))

    summary_by_study = {}
    for study in studies:
        study_tasks = [row for row in tasks if row["study"] == study]
        study_targets = {
            key: value for key, value in target_rows.items() if key[0] == study
        }
        summary_by_study[study] = {
            "n_tasks": len(study_tasks),
            "n_target_clusters": len(study_targets),
            "family_lower_violations": sum(
                int(float(row["diversity_lower"]) > float(row["family_rms_error"]) + TOL)
                for row in study_tasks
            ),
            "worst_lower_violations": sum(
                int(float(row["diameter_lower"]) > float(row["worst_member_error"]) + TOL)
                for row in study_tasks
            ),
            "family_upper_tasks_covered": sum(
                int(float(row["family_rms_error"]) <= float(row["family_upper"]) + TOL)
                for row in study_tasks
            ),
            "family_upper_targets_covered": sum(
                int(value["family_covered"]) for value in study_targets.values()
            ),
            "worst_upper_targets_covered": sum(
                int(value["worst_covered"]) for value in study_targets.values()
            ),
        }

    reported_studies = {row["study"]: row for row in studies_reported}
    audit.check(
        "study_key_set_matches",
        set(summary_by_study) == set(reported_studies),
        studies,
    )
    integer_fields = [
        "n_tasks",
        "n_target_clusters",
        "family_lower_violations",
        "worst_lower_violations",
        "family_upper_tasks_covered",
        "family_upper_targets_covered",
        "worst_upper_targets_covered",
    ]
    for study, computed in summary_by_study.items():
        for field in integer_fields:
            audit.check(
                "study_{}_{}".format(study, field),
                computed[field] == int(reported_studies[study][field]),
                "computed={} reported={}".format(
                    computed[field], reported_studies[study][field]
                ),
            )

    metrics = {
        "n_studies": len(studies),
        "n_evaluation_tasks": len(tasks),
        "n_target_clusters": len(target_rows),
        "family_lower_violations": family_lower_violations,
        "worst_lower_violations": worst_lower_violations,
        "family_upper_tasks_covered": family_upper_tasks_covered,
        "family_upper_task_coverage": family_upper_tasks_covered / len(tasks),
        "family_upper_targets_covered": family_upper_targets_covered,
        "family_upper_target_coverage": family_upper_targets_covered / len(target_rows),
        "worst_upper_targets_covered": worst_upper_targets_covered,
        "worst_upper_target_coverage": worst_upper_targets_covered / len(target_rows),
        "max_hilbert_identity_absolute_residual": max_identity_residual,
    }

    integer_status_fields = [
        "n_studies",
        "n_evaluation_tasks",
        "n_target_clusters",
        "family_lower_violations",
        "worst_lower_violations",
        "family_upper_tasks_covered",
        "family_upper_targets_covered",
        "worst_upper_targets_covered",
    ]
    float_status_fields = [
        "family_upper_task_coverage",
        "family_upper_target_coverage",
        "worst_upper_target_coverage",
        "max_hilbert_identity_absolute_residual",
    ]
    for field in integer_status_fields:
        audit.check(
            "run_status_{}".format(field),
            metrics[field] == int(run_status[field]),
            "computed={} reported={}".format(metrics[field], run_status[field]),
        )
    for field in float_status_fields:
        audit.check(
            "run_status_{}".format(field),
            close(metrics[field], run_status[field]),
            "computed={} reported={}".format(metrics[field], run_status[field]),
        )

    e182_target_rows = {
        key: value for key, value in target_rows.items() if key[0] == "E182_GSE225807"
    }
    e182_family_covered = sum(
        int(value["family_covered"]) for value in e182_target_rows.values()
    )
    e182_worst_covered = sum(
        int(value["worst_covered"]) for value in e182_target_rows.values()
    )
    audit.check("e182_status_remains_fail", e182_status["status"] == "FAIL", e182_status["status"])
    audit.check("e182_targets_20", len(e182_target_rows) == 20, len(e182_target_rows))
    audit.check("e182_family_covered_16", e182_family_covered == 16, e182_family_covered)
    audit.check("e182_worst_covered_19", e182_worst_covered == 19, e182_worst_covered)
    audit.check(
        "e182_registered_gate_failed",
        e182_status["gates"]["target_simultaneous_coverage_at_least_0_85"] is False,
        e182_status["gates"]["target_simultaneous_coverage_at_least_0_85"],
    )

    beta_reference = beta_binomial_cdf(16, 20, 18, 2)
    audit.check(
        "e182_beta_binomial_reference",
        close(
            beta_reference,
            run_status["e182_beta_binomial_reference_probability_k_at_most_16"],
        ),
        beta_reference,
    )

    return {
        "metrics": metrics,
        "study_summary": summary_by_study,
        "e182": {
            "status": e182_status["status"],
            "targets": len(e182_target_rows),
            "family_upper_targets_covered": e182_family_covered,
            "worst_upper_targets_covered": e182_worst_covered,
            "beta_binomial_reference_probability_k_at_most_16": beta_reference,
        },
        "provenance": {
            "e181_manifest_entries_verified": e181_manifest_entries,
            "e183_input_hash_entries_verified": len(e183_inputs),
            "raw_expression_or_truth_arrays_read": 0,
        },
    }


def write_outputs(output_dir, audit, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "safeconf_current_release_validation_v1",
        "status": "PASS" if audit.passed else "FAIL",
        "checks_total": len(audit.checks),
        "checks_failed": sum(not item["passed"] for item in audit.checks),
        **result,
        "checks": audit.checks,
    }
    with (output_dir / "VALIDATION.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    with (output_dir / "CURRENT_RELEASE_MAIN_NUMBERS.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "study",
                "n_tasks",
                "n_target_clusters",
                "family_lower_violations",
                "worst_lower_violations",
                "family_upper_tasks_covered",
                "family_upper_targets_covered",
                "worst_upper_targets_covered",
            ],
        )
        writer.writeheader()
        for study in sorted(result["study_summary"]):
            writer.writerow({"study": study, **result["study_summary"][study]})

    m = result["metrics"]
    e182 = result["e182"]
    lines = [
        "# SafeConf current release validation",
        "",
        "Status: **{}**".format("PASS" if audit.passed else "FAIL"),
        "",
        "This audit used committed certificate tables only. It did not read raw "
        "expression matrices, evaluation truth arrays, or model checkpoints.",
        "",
        "## Recomputed headline values",
        "",
        "- Studies: `{}`".format(m["n_studies"]),
        "- Evaluation tasks: `{}`".format(m["n_evaluation_tasks"]),
        "- Target clusters: `{}`".format(m["n_target_clusters"]),
        "- Family / worst-member lower-bound violations: `{}` / `{}`".format(
            m["family_lower_violations"], m["worst_lower_violations"]
        ),
        "- Family upper task coverage: `{}/{} = {:.2%}`".format(
            m["family_upper_tasks_covered"],
            m["n_evaluation_tasks"],
            m["family_upper_task_coverage"],
        ),
        "- Family upper target coverage: `{}/{} = {:.2%}`".format(
            m["family_upper_targets_covered"],
            m["n_target_clusters"],
            m["family_upper_target_coverage"],
        ),
        "- Worst-member upper target coverage: `{}/{} = {:.2%}`".format(
            m["worst_upper_targets_covered"],
            m["n_target_clusters"],
            m["worst_upper_target_coverage"],
        ),
        "- Maximum Hilbert identity residual: `{:.3e}`".format(
            m["max_hilbert_identity_absolute_residual"]
        ),
        "",
        "## E182 lock",
        "",
        "- Registered status remains: **{}**".format(e182["status"]),
        "- Family upper target coverage: `{}/{} = {:.1%}`".format(
            e182["family_upper_targets_covered"],
            e182["targets"],
            e182["family_upper_targets_covered"] / e182["targets"],
        ),
        "- Beta-binomial reference `P(K <= 16)`: `{:.6f}`".format(
            e182["beta_binomial_reference_probability_k_at_most_16"]
        ),
        "- This reference probability does not change the preregistered FAIL.",
        "",
        "## Integrity",
        "",
        "- Checks run: `{}`".format(len(audit.checks)),
        "- Checks failed: `{}`".format(
            sum(not item["passed"] for item in audit.checks)
        ),
        "- E181 manifest entries verified: `{}`".format(
            result["provenance"]["e181_manifest_entries_verified"]
        ),
        "- E183 input hash entries verified: `{}`".format(
            result["provenance"]["e183_input_hash_entries_verified"]
        ),
        "",
    ]
    (output_dir / "VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    audit = Audit()
    result = recompute_release(repo, audit)
    payload = write_outputs(output_dir, audit, result)
    print(
        "SafeConf release validation: {} ({} checks, {} failed)".format(
            payload["status"], payload["checks_total"], payload["checks_failed"]
        )
    )
    print("Report: {}".format(output_dir / "VALIDATION_REPORT.md"))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == "__main__":
    main()
