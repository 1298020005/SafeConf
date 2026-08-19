#!/bin/bash
# SafeConf 数据集状况
#
# 经确认：所有核心数据集已在服务器上，无需下载。
# 来源: scPerturBench (Zenodo DOI: 10.5281/zenodo.14607156)
#        scPerturb (Zenodo DOI: 10.5281/zenodo.13350497)
#
# 如果需要重新下载（例如文件损坏），取消注释对应的 wget 行。

DEST_DIR="/home/yyf/data/singlecell_perturbation_atlas"
EXTRA_DIR="${DEST_DIR}/extra_official/cellular_context_generalization"
SCPERTURB_DIR="${DEST_DIR}/official_scperturb"

echo "========================================="
echo "SafeConf 数据集可用性检查"
echo "========================================="
echo ""

# ============================================
# 已有数据集（确认在磁盘上）
# ============================================

echo "--- 已有数据集检查 ---"
echo ""

DATASETS_P1=(
    "${EXTRA_DIR}/kangCrossCell.h5ad"
    "${EXTRA_DIR}/kangCrossPatient.h5ad"
)

DATASETS_P2=(
    "${EXTRA_DIR}/TCDD.h5ad"
    "${EXTRA_DIR}/sciplex3.h5ad"
)

DATASETS_EXISTING=(
    "${EXTRA_DIR}/Haber.h5ad"
    "${EXTRA_DIR}/Parekh.h5ad"
    "${EXTRA_DIR}/KaggleCrossCell.h5ad"
    "${EXTRA_DIR}/KaggleCrossPatient.h5ad"
    "${EXTRA_DIR}/crossPatient.h5ad"
    "${DEST_DIR}/official_generalization/Frangieh.h5ad"
)

DATASETS_P3=(
    "${SCPERTURB_DIR}/ShifrutMarson2018.h5ad"
    "${SCPERTURB_DIR}/AissaBenevolenskaya2021.h5ad"
    "${SCPERTURB_DIR}/LaraAstiasoHuntly2023_exvivo.h5ad"
    "${SCPERTURB_DIR}/LaraAstiasoHuntly2023_invivo.h5ad"
)

echo "[已用数据集]"
for f in "${DATASETS_EXISTING[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  ✅ $(basename $f) ($SIZE)"
    else
        echo "  ❌ $(basename $f) - MISSING!"
    fi
done
echo ""

echo "[P1: 核心扩展 - gene_main]"
for f in "${DATASETS_P1[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  ✅ $(basename $f) ($SIZE)"
    else
        echo "  ❌ $(basename $f) - MISSING! 需要下载"
    fi
done
echo ""

echo "[P2: 强烈推荐 - chem_robust]"
for f in "${DATASETS_P2[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  ✅ $(basename $f) ($SIZE)"
    else
        echo "  ❌ $(basename $f) - MISSING! 需要下载"
    fi
done
echo ""

echo "[P3: 可选补充]"
for f in "${DATASETS_P3[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  ✅ $(basename $f) ($SIZE)"
    else
        echo "  ⚠️  $(basename $f) - 未找到（可选下载）"
    fi
done
echo ""

# ============================================
# 下载命令（取消注释以使用）
# ============================================

# --- P1: kangCrossCell (21.6 MB gz) ---
# wget --continue -O "${EXTRA_DIR}/kangCrossCell.h5ad.gz" \
#     "https://zenodo.org/api/records/14607156/files/kangCrossCell.h5ad.gz/content"
# gunzip -k "${EXTRA_DIR}/kangCrossCell.h5ad.gz"

# --- P1: kangCrossPatient (20.8 MB gz) ---
# wget --continue -O "${EXTRA_DIR}/kangCrossPatient.h5ad.gz" \
#     "https://zenodo.org/api/records/14607156/files/kangCrossPatient.h5ad.gz/content"
# gunzip -k "${EXTRA_DIR}/kangCrossPatient.h5ad.gz"

# --- P2: TCDD (186.7 MB gz) ---
# wget --continue -O "${EXTRA_DIR}/TCDD.h5ad.gz" \
#     "https://zenodo.org/api/records/14607156/files/TCDD.h5ad.gz/content"
# gunzip -k "${EXTRA_DIR}/TCDD.h5ad.gz"

# --- P2: sciplex3 (26.2 MB gz) ---
# wget --continue -O "${EXTRA_DIR}/sciplex3.h5ad.gz" \
#     "https://zenodo.org/api/records/14607156/files/sciplex3.h5ad.gz/content"
# gunzip -k "${EXTRA_DIR}/sciplex3.h5ad.gz"

# --- P3: ShifrutMarson2018 (831 MB, no compression) ---
# wget --continue -O "${SCPERTURB_DIR}/ShifrutMarson2018.h5ad" \
#     "https://zenodo.org/api/records/13350497/files/ShifrutMarson2018.h5ad/content"

# --- P3: AissaBenevolenskaya2021 (43.8 MB) ---
# wget --continue -O "${SCPERTURB_DIR}/AissaBenevolenskaya2021.h5ad" \
#     "https://zenodo.org/api/records/13350497/files/AissaBenevolenskaya2021.h5ad/content"

# --- P3: LaraAstiasoHuntly2023_exvivo (1030 MB) ---
# wget --continue -O "${SCPERTURB_DIR}/LaraAstiasoHuntly2023_exvivo.h5ad" \
#     "https://zenodo.org/api/records/13350497/files/LaraAstiasoHuntly2023_exvivo.h5ad/content"

echo "========================================="
echo "检查完成。"
echo ""
echo "下一步: 运行 SafeConf eligibility audit"
echo "  python -m safetrans_confidence.cli.run_eligibility_audit \\"
echo "    --atlas-root ${DEST_DIR}"
echo "========================================="
