# -*- coding: utf-8 -*-
"""11/12 · 查询端：照片 -> OSNet特征 -> 与档案库比对 -> 融合排序 -> Top-k

search_core 只吃"已归一化向量+档案库字典"，与特征来源解耦：
现在用 Market 真实图/档案库crop验证，实拍素材到位后换数据源即可。
"""
import os
import sys
import glob
import importlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))        # fusion
sys.path.insert(0, HERE)                             # 07 同目录

from fusion import fuse_max, fuse_mean, topk_rank    # noqa: E402
m07 = importlib.import_module("07_market_eval")      # noqa: E402

INDEX_NPZ = os.path.join(ROOT, "data", "index", "index.npz")


def load_index(npz_path=INDEX_NPZ):
    """{track_id: {'emb':(K,512), 'frames':(K,), 'paths':[K]}}，并校验向量已归一化。"""
    z = np.load(npz_path, allow_pickle=True)
    tids = sorted({int(k[4:]) for k in z.files if k.startswith("emb_")})
    db = {}
    for tid in tids:
        embs = z[f"emb_{tid}"]
        assert np.allclose((embs ** 2).sum(1), 1.0, atol=1e-3), f"ID{tid} 未归一化"
        db[tid] = {"emb": embs, "frames": z[f"frames_{tid}"],
                  "paths": [str(p) for p in z[f"paths_{tid}"]]}
    return db


def search_core(query_embs, db, fusion="max", k=5):
    """query_embs (n,512) np 归一化。返回 [(track_id, score, frames, 代表crop, 最佳证据向量号)]"""
    q = np.asarray(query_embs, dtype=np.float64)
    assert q.ndim == 2 and q.shape[1] == 512
    tids = sorted(db)
    # 每人一块：算 (n, K_person) 相似度 -> 该人融合分 + 最佳证据(crop)索引
    results = []
    for tid in tids:
        s_mat = q @ db[tid]["emb"].T                    # (n, K)
        # max融合: 任一查询×任一留底的最大; mean融合: 全体均值
        score = s_mat.max() if fusion == "max" else s_mat.mean()
        best_evi = int(s_mat.argmax(axis=1)[0])          # 第1张查询图的最佳向量
        results.append((tid, float(score),
                        [int(f) for f in db[tid]["frames"]],
                        db[tid]["paths"][best_evi], best_evi))
    results.sort(key=lambda r: -r[1])
    return results[:k]


def embed_query_images(model, image_paths):
    """复用07 extract_feats：N 张查询图 -> (N,512) 归一化向量。"""
    return m07.extract_feats(model, [(p, 0, 0) for p in image_paths]).numpy()


def main():
    db = load_index()
    print(f"档案库: {len(db)} 个ID")
    model = m07.build_reid_model()

    # 自查询冒烟：每个ID拿自己的一张crop当查询图（应命中自身、且分数最高）
    qpaths = [db[tid]["paths"][0] for tid in sorted(db)]
    qembs = embed_query_images(model, qpaths)

    n_hit = 0
    for i, tid_true in enumerate(sorted(db)):
        res = search_core(qembs[i:i + 1], db, fusion="max", k=5)
        rank = next((r + 1 for r, x in enumerate(res) if x[0] == tid_true), None)
        hit = "HIT" if rank == 1 else f"rank={rank}"
        n_hit += (rank == 1)
        print(f"查询自 ID{tid_true:02d}: {hit:8s} "
              f"top1=ID{res[0][0]:02d} score={res[0][1]:.4f}")
    print(f"\n自查询 Top-1 命中率: {n_hit}/{len(qpaths)} = {n_hit / len(qpaths):.0%}")
    print("（每张查询图取自档案库自身crop，命中自身是最基本要求；"
          "跨图/多角度命中率待实拍或PRW素材，见12消融设计）")


if __name__ == "__main__":
    main()
