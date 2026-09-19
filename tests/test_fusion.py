# -*- coding: utf-8 -*-
"""12 · fusion 单元测试：pytest tests/test_fusion.py -v"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from fusion import fuse_max, fuse_mean, topk_rank


def test_max_picks_rowwise_max():
    m = [[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]]
    assert np.allclose(fuse_max(np.array(m)), [0.8, 0.9])


def test_mean_equals_average():
    m = [[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]]
    assert np.allclose(fuse_mean(np.array(m)), [0.46666667, 0.53333333])


def test_single_image_fusion_degenerates_to_itself():
    """1 张查询图时，max/mean 都等于原始相似度——消融表的基准行。"""
    m = [[0.3, 0.7, 0.1]]
    assert np.allclose(fuse_max(np.array(m)), m[0])
    assert np.allclose(fuse_mean(np.array(m)), m[0])


def test_max_robust_to_blurry_query():
    """3 张查询里混 1 张糊图（与所有人相似度都低）：max 不受影响，mean 被拉低。"""
    good = [[0.9, 0.4], [0.85, 0.5]]      # 两张清晰图：目标A 像
    blurry = [[0.3, 0.35]]                 # 一张糊图
    mA = fuse_max(np.array(good + blurry))
    meanA = fuse_mean(np.array(good + blurry))
    mA2 = fuse_max(np.array(good))         # 没糊图时
    assert mA[0] == mA2[0]                 # max: 糊图完全不影响
    assert meanA[0] < mA2[0]               # mean: 被糊图拉低


def test_mean_can_beaten_by_majority_evidence():
    """多张"中等像"在 mean 下可反超单张"极像"——两策略分歧点，消融要展示。"""
    m = [[0.95, 0.6], [0.2, 0.62], [0.2, 0.61]]   # A 一张爆高其余很低; B 三张稳定中等
    assert fuse_max(np.array(m))[0] > fuse_max(np.array(m))[1]   # max: A赢(0.95)
    assert fuse_mean(np.array(m))[1] > fuse_mean(np.array(m))[0] # mean: B赢(0.61>0.45)


def test_topk_rank_order():
    ids = ["p3", "p1", "p2"]
    out = topk_rank(np.array([0.2, 0.9, 0.5]), ids, k=2)
    assert out[0][0] == "p1" and out[1][0] == "p2" and len(out) == 2


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        fuse_max(np.array([0.1, 0.2]))     # 1维
    with pytest.raises(ValueError):
        fuse_mean(np.zeros((0, 3)))        # 空
