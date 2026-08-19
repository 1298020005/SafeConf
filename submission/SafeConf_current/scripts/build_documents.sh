#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v pandoc >/dev/null 2>&1; then
  PANDOC="$(command -v pandoc)"
elif [[ -x /home/yyf/.local/opt/pandoc/bin/pandoc ]]; then
  PANDOC=/home/yyf/.local/opt/pandoc/bin/pandoc
else
  echo "pandoc was not found. Install pandoc 3.x or place it at /home/yyf/.local/opt/pandoc/bin/pandoc." >&2
  exit 1
fi

cd "$HERE"

"$PANDOC" SafeConf_manuscript.md \
  --from=markdown+tex_math_dollars+tex_math_single_backslash \
  --standalone \
  --citeproc \
  --resource-path=. \
  --bibliography=references.bib \
  --csl=styles/biomed-central.csl \
  --metadata=reference-section-title:References \
  -o SafeConf_manuscript.docx

"$PANDOC" SafeConf_supplement.md \
  --from=markdown+tex_math_dollars+tex_math_single_backslash \
  --standalone \
  --citeproc \
  --resource-path=. \
  --bibliography=references.bib \
  --csl=styles/biomed-central.csl \
  --metadata=reference-section-title:References \
  -o SafeConf_supplement.docx

"$PANDOC" Cover_letter_BMC_Bioinformatics.md \
  --from=markdown \
  --standalone \
  -o Cover_letter_BMC_Bioinformatics.docx

python scripts/postprocess_word.py

python -m zipfile -c SafeConf_source_data.zip \
  tables/Table_1_study_design.csv \
  tables/Table_2_certificate_results.csv \
  tables/Table_S1_family_comparisons.csv \
  tables/Table_S2_gse225807_targets.csv \
  tables/Table_S3_gse225807_tasks.csv \
  tables/Table_S4_difficulty_setting_summary.csv \
  tables/Table_S5_difficulty_macro_bootstrap.csv \
  tables/Table_S6_cross_dataset_summary.csv \
  tables/Table_S7_difficulty_task_certificates.csv \
  tables/Table_S8_cross_dataset_task_certificates.csv \
  tables/Table_S9_model_error_concordance.csv \
  tables/Table_S10_score_model_specificity.csv \
  tables/Table_S11_nested_upper_baselines.csv \
  tables/Table_S12_prescribe_native_endpoint.csv \
  tables/Table_S13_prescribe_incremental.csv \
  tables/Table_S14_prescribe_redundancy.csv

if command -v libreoffice >/dev/null 2>&1; then
  LO_PROFILE="${TMPDIR:-/tmp}/safeconf_libreoffice_profile"
  mkdir -p "$LO_PROFILE"
  libreoffice \
    "-env:UserInstallation=file://$LO_PROFILE" \
    --headless \
    --convert-to pdf \
    --outdir "$HERE" \
    SafeConf_manuscript.docx \
    SafeConf_supplement.docx \
    Cover_letter_BMC_Bioinformatics.docx >/dev/null
fi

echo "Built SafeConf_manuscript.docx"
echo "Built SafeConf_supplement.docx"
echo "Built Cover_letter_BMC_Bioinformatics.docx"
