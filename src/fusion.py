# -*- coding: utf-8 -*-
"""12 · 多图查询融合策略（毕设创新点核心，纯 numpy，可单测）

约定：sim_mat 形状 (n_query_imgs, n_gallery_items)，元素为余弦相似度∈[-1,1]。
融合输出每人一个标量分数 (n_gallery_items,)。
"""
import numpy as np


def fuse_max(sim_mat: np.ndarray) -> np.ndarray:
    """max 融合：每人取【任一向量×任一查询图】的最高相似度。

    语义：只要档案里任何一张留底和任何一张查询图非常像，就强烈支持命中。
    适合查询图角度/光照差异大的场景（总有一张对得上）。
    """
    sim_mat = np.asarray(sim_mat, dtype=np.float64)
    if sim_mat.ndim != 2 or 0 in sim_mat.shape:
        raise ValueError(f"sim_mat 需为非空2维矩阵, got {sim_mat.shape}")
    return sim_mat.max(axis=0)


def fuse_mean(sim_mat: np.ndarray) -> np.ndarray:
    """mean 融合：对每人所有相似度取平均。

    语义：要求所有匹配证据整体像；被单张糊图拉低。对照组用。
    """
    sim_mat = np.asarray(sim_mat, dtype=np.float64)
    if sim_mat.ndim != 2 or 0 in sim_mat.shape:
        raise ValueError(f"sim_mat 需为非空2维矩阵, got {sim_mat.shape}")
    return sim_mat.mean(axis=0)


def topk_rank(scores: np.ndarray, ids: list, k: int = 5) -> list:
    """返回 [(id, score), ...] 按分数降序 Top-k。"""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    return [(ids[j], float(scores[j])) for j in order[:k]]
