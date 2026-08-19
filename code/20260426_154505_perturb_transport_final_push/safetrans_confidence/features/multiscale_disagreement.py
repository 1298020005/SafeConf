"""Multi-Scale Disagreement Features（多尺度分歧特征）

从两个 predictor（预测器）的 predicted_effect（预测效应向量）中，
提取 8 种不同维度的 disagreement（分歧）信号。

输入：两个 numpy array，每个 5000 维（代表 5000 个基因的预测变化量）
输出：一个 dict（字典），包含 8 个 float（浮点数）特征

这 8 种特征分别从不同角度衡量"两个预测器有多不一样"：
1. disagree_rmse: 整体距离（均方根误差）
2. disagree_direction_agreement: 方向一致率（上调/下调是否一致）
3. disagree_rank_spearman: 排名一致性（哪些基因变化最大的排序是否一致）
4. disagree_topk_jaccard: Top-K 基因重叠（最重要的 K 个基因是否重合）
5. disagree_sparsity_jaccard: 稀疏模式分歧（哪些基因有变化 vs 没变化的判断是否一致）
6. disagree_cosine_distance: 余弦距离（预测趋势形状是否相似）
7. disagree_variance_ratio: 方差比（波动程度差异）
8. disagree_tail_divergence: 尾部差异（变化最大的 top-5% 基因的具体数值差异）
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# 所有 8 种多尺度分歧特征的列名前缀
MULTISCALE_DISAGREEMENT_COLS = [
    "disagree_rmse",
    "disagree_direction_agreement",
    "disagree_rank_spearman",
    "disagree_topk_jaccard",
    "disagree_sparsity_jaccard",
    "disagree_cosine_distance",
    "disagree_variance_ratio",
    "disagree_tail_divergence",
]


def compute_multiscale_disagreement(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    top_k: int = 50,
) -> dict[str, float]:
    """计算 8 种分歧特征。

    Parameters
    ----------
    vec_a : np.ndarray
        预测器 A 的预测效应向量（5000 个基因的变化量）。
    vec_b : np.ndarray
        预测器 B 的预测效应向量。
    top_k : int, default 50
        Top-K 基因数量，用于 topk_jaccard 特征。

    Returns
    -------
    dict[str, float]
        包含 8 个分歧特征的字典，键名以 "disagree_" 开头。

    Examples
    --------
    >>> import numpy as np
    >>> a = np.random.randn(5000)
    >>> b = a + np.random.randn(5000) * 0.1
    >>> result = compute_multiscale_disagreement(a, b)
    >>> sorted(result.keys())
    ['disagree_cosine_distance', 'disagree_direction_agreement', ...]
    """
    # 确保输入是一维 float 数组
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()

    if a.shape != b.shape:
        raise ValueError(
            f"vec_a and vec_b must have the same shape, got {a.shape} vs {b.shape}"
        )

    results: dict[str, float] = {}

    # ---- 特征 1: disagree_rmse（整体距离）----
    # 两个向量每个位置差值的平方，取平均，再开方
    # 就像问"两个医生的评分平均差多少分"
    # 值越大 → 两个预测器整体差距越大 → 越不可信
    results["disagree_rmse"] = float(np.sqrt(np.mean((a - b) ** 2)))

    # ---- 特征 2: disagree_direction_agreement（方向一致率）----
    # 对每个基因，两个预测器是否同意"该上调（变大）还是下调（变小）"
    # 只看两个预测器都认为"有变化"的基因（排除都是 0 的）
    sign_a = np.sign(a)  # +1 = 上调，-1 = 下调，0 = 没变
    sign_b = np.sign(b)
    both_active = (sign_a != 0) & (sign_b != 0)
    if both_active.sum() > 0:
        # 方向一致的比例：1.0 = 完全一致，0.0 = 完全相反
        results["disagree_direction_agreement"] = float(
            np.mean(sign_a[both_active] == sign_b[both_active])
        )
    else:
        results["disagree_direction_agreement"] = float("nan")

    # ---- 特征 3: disagree_rank_spearman（排名一致性）----
    # 两个预测器对"哪些基因变化最大"的排名是否一致
    # Spearman rho（斯皮尔曼排序相关系数）：+1 = 排名完全一致，0 = 无关，-1 = 完全相反
    if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0:
        rho, _ = stats.spearmanr(a, b)
        results["disagree_rank_spearman"] = float(rho)
    else:
        results["disagree_rank_spearman"] = float("nan")

    # ---- 特征 4: disagree_topk_jaccard（Top-K 基因重叠）----
    # 各自选出"变化最大的 K 个基因"，看交集占并集的比例
    # Jaccard（杰卡德系数）：交集大小 / 并集大小，0 = 完全不重合，1 = 完全重合
    k = min(top_k, len(a) // 2)
    if k > 0:
        top_a = set(np.argsort(np.abs(a))[-k:].tolist())
        top_b = set(np.argsort(np.abs(b))[-k:].tolist())
        union = top_a | top_b
        results["disagree_topk_jaccard"] = (
            len(top_a & top_b) / len(union) if union else float("nan")
        )
    else:
        results["disagree_topk_jaccard"] = float("nan")

    # ---- 特征 5: disagree_sparsity_jaccard（稀疏模式分歧）----
    # 两个预测器对"哪些基因有变化 vs 哪些没变化"是否一致
    # 用中位数绝对值作为"有变化"的阈值
    diff_abs = np.abs(a - b)
    threshold = float(np.median(diff_abs))
    if threshold > 0:
        active_a = np.abs(a) > threshold
        active_b = np.abs(b) > threshold
        union_mask = active_a | active_b
        union_count = int(np.sum(union_mask))
        intersect_count = int(np.sum(active_a & active_b))
        results["disagree_sparsity_jaccard"] = (
            float(intersect_count / max(union_count, 1))
        )
    else:
        results["disagree_sparsity_jaccard"] = float("nan")

    # ---- 特征 6: disagree_cosine_distance（余弦距离）----
    # 两个向量的"方向"差多少（不管长度）
    # cosine similarity（余弦相似度）= 1 表示方向完全一致
    # distance = 1 - similarity，范围 [0, 2]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 1e-12 and norm_b > 1e-12:
        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        results["disagree_cosine_distance"] = 1.0 - cos_sim
    else:
        results["disagree_cosine_distance"] = float("nan")

    # ---- 特征 7: disagree_variance_ratio（方差比）----
    # 一个预测的波动大，另一个的波动小？
    # 用 log ratio（对数比）让比值对称：log(1) = 0 表示一样大
    var_a = float(np.var(a))
    var_b = float(np.var(b))
    if var_a > 1e-12 and var_b > 1e-12:
        results["disagree_variance_ratio"] = abs(np.log(var_a / var_b))
    else:
        results["disagree_variance_ratio"] = float("nan")

    # ---- 特征 8: disagree_tail_divergence（尾部差异）----
    # 只看"变化最大的 top-5% 基因"，它们的数值差多少
    # 这些基因往往是生物学上最重要的
    n_tail = max(1, int(0.05 * len(a)))
    avg_abs = (np.abs(a) + np.abs(b)) / 2.0
    tail_idx = np.argsort(avg_abs)[-n_tail:]
    results["disagree_tail_divergence"] = float(
        np.sqrt(np.mean((a[tail_idx] - b[tail_idx]) ** 2))
    )

    return results
