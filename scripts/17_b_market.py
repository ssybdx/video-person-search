# -*- coding: utf-8 -*-
"""17 · B 组消融（Market 多查询正解版）：单图/3图/5图 × max/mean/top3mean

背景：校服素材 B 组（scripts/16_b_group_ablation.py）因查询图 18 秒连拍同质 +
统一校服 + 样本量 4，测不出"多图优于单图"——已按用户要求作为反例入档。
本脚本用 Market-1501 重做：query 集 3368 张 / 750 个身份，人均 4.5 张、
跨相机跨姿态——"同一人的真正多图"在这里天然存在，max 融合收益可直接测量。

协议 v2（图像级 CMC + 官方剔同相机）：
  v1 教训：把分数聚合到"身份级"再排序，同相机的连拍近重复帧会替目标身份霸榜，
  九格全 100%——这正是 07B-1"99.61% 虚高"的同款陷阱重演，故作废改协议。
- gallery = bounding_box_test：剔 junk（pid=-1）、**保留 distractor（pid=0）当困难负样本**
- 每个查询身份 pid 取前 k 张（k ∈ {1,3,5}，文件名序=时间序）；主表只用 n>=5 身份
  （三档同一人群，公平对比），附表报全量
- **剔同相机**：剔除 gallery 中 camid 属于本组查询相机集合的全部图；
  剔后该身份若还有图才参评（没有则跳过，与官方一致）
- 图像级融合打分（对每张 gallery 图 j 算一个分）：
    max       = max_i(q_i · g_j)        任一查询命中即证据
    mean      = q̄ · g_j                查询向量先平均（朴素基线）
    top3mean  = top3_i(q_i · g_j).mean() 前3平均，折中
- hit@1 = 排序第 1 张是目标身份图；hit@3 = 前 3 内含目标图
  mAP = 以目标身份全部图为正例的 Average Precision
输出：终端矩阵 + docs\market_multiquery_b.csv
"""
import glob
import importlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
m07 = importlib.import_module("07_market_eval")   # 复用 parse_name / extract_feats / build_reid_model

MK = os.path.join(ROOT, "data", "market1501", "Market-1501-v15.09.15")
CSV_OUT = os.path.join(ROOT, "docs", "market_multiquery_b.csv")
KS = (1, 3, 5)
FUSIONS = ("max", "mean", "top3mean")


def collect_queries():
    """{pid: [(path, camid), ...]} 按文件名序；剔 pid<=0 异常。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(MK, "query", "*.jpg"))):
        pid, cam = m07.parse_name(p)
        if pid <= 0:
            continue
        out.setdefault(pid, []).append((p, cam))
    return out


def collect_gallery():
    """[(path, pid, camid)]，剔 junk；保留 distractor(pid=0) 当困难负样本。"""
    g = []
    for p in sorted(glob.glob(os.path.join(MK, "bounding_box_test", "*.jpg"))):
        pid, cam = m07.parse_name(p)
        if pid == -1:
            continue
        g.append((p, pid, cam))
    return g


def fuse_scores(s, fusion):
    """s: (k, M) 余弦 -> (M,) 每张 gallery 图的融合分"""
    if fusion == "max":
        return s.max(axis=0)
    if fusion == "mean":
        return s.mean(axis=0)
    k = min(3, s.shape[0])
    return np.partition(s, -k, axis=0)[-k:].mean(axis=0)     # top3mean


def score_group(s, pids_k, tpid, fusion):
    """返回该融合策略的 (hit1, hit3, ap)。"""
    v = fuse_scores(s, fusion)
    order = np.argsort(-v)
    is_t = np.array([p == tpid for p in pids_k])
    rank = int(np.argmax(is_t[order])) + 1                   # 第一张目标图的名次
    hit1, hit3 = rank == 1, rank <= 3
    # AP：按排序累计 precision 在目标位置上的平均（截断到 top50 与官方近似）
    prec, n_hit = 0.0, 0
    for i in range(min(50, len(order))):
        if is_t[order[i]]:
            n_hit += 1
            prec += n_hit / (i + 1)
    total_t = max(1, int(is_t.sum()))
    ap = prec / min(total_t, 50)
    return hit1, hit3, ap


def main():
    t0 = time.time()
    queries, galleries = collect_queries(), collect_gallery()
    g_pids = [p for _, p, _ in galleries]
    g_cams = [c for _, _, c in galleries]
    print(f"query {sum(len(v) for v in queries.values())} 张 / {len(queries)} 身份；"
          f"gallery {len(galleries)} 张（distractor {g_pids.count(0)}）")

    model = m07.build_reid_model()                            # 自训 A2（07B-2 冠军）
    gemb = m07.extract_feats(model, galleries).cpu().numpy().astype(np.float64)
    print(f"gallery 特征完成 {gemb.shape}（{time.time()-t0:.0f}s）")

    q5 = {p: v[:5] for p, v in queries.items() if len(v) >= 5}
    q3 = {p: v[:3] for p, v in queries.items() if len(v) >= 3}
    print(f"身份子集：n>=5 {len(q5)} 个 | n>=3 {len(q3)} 个")

    feats_cache = {}

    def qemb_of(pid, k):
        if pid not in feats_cache:
            paths = [p for p, _ in queries[pid][:5]]
            feats_cache[pid] = m07.extract_feats(
                model, [(p, pid, 0) for p in paths]).cpu().numpy().astype(np.float64)
        return feats_cache[pid][:k]

    g_pids_arr = np.asarray(g_pids)

    def run_grid(label, pop, k_list):
        """v4 公平网格：剔相机集 = 该身份【参评查询全集】相机的并集（与 k 无关），
        剔后目标无图的身份【整格跳过】——保证同一张表里各 k 档、各策略人群完全一致。
        v2/v3 教训：v2 pid 级打分被同相机白送分打满(100%)；v3 剔相机集按 k 变，
        导致 1图 n=412 与 5图 n=94 人群不同，行间不可比——判卷协议再次先于数字。"""
        stats = {f"{k}图-{fu}": [] for k in k_list for fu in FUSIONS}
        kept_ids = 0
        for pid in sorted(pop):
            qs = pop[pid]
            qcams = {c for _, c in qs}                       # 并集：与 k 无关
            keep = np.array([c not in qcams for c in g_cams])
            pids_k = list(g_pids_arr[keep])
            if pid not in pids_k:
                continue                                      # 剔后目标无图：整身份跳过
            kept_ids += 1
            for k in k_list:
                if len(qs) < k:
                    continue
                s = qemb_of(pid, k) @ gemb[keep].T
                for fu in FUSIONS:
                    stats[f"{k}图-{fu}"].append(score_group(s, pids_k, pid, fu))
            if len(feats_cache) > 150:
                feats_cache.clear()
        print(f"\n=== {label}（参评身份 {kept_ids}）===")
        print(f"{'策略':<13} {'hit@1':<9} {'hit@3':<9} {'mAP':<9} 试验数")
        out = []
        for name, lst in stats.items():
            if not lst:
                continue
            a = np.array(lst)
            k, fu = name.split("-")
            r = (label, k, fu, float(a[:, 0].mean()), float(a[:, 1].mean()),
                 float(a[:, 2].mean()), len(lst))
            print(f"{name:<13} {r[3]:<9.2%} {r[4]:<9.2%} {r[5]:<9.2%} {r[6]}")
            out.append(r)
        return out

    rows = []
    rows += run_grid("表A 多图公平表(n>=5)", q5, [1, 3, 5])     # 主结论表
    rows += run_grid("表B 较大样本(n>=3)", q3, [1, 3])          # 1→3 效应的扩样验证

    # 参考行：官方单查询口径（剔 1 相机）下的 1图-max，对照 07B-2 量级
    ref = []
    for pid in sorted(q5):
        c1 = q5[pid][0][1]
        keep = np.array([c != c1 for c in g_cams])
        pids_k = list(g_pids_arr[keep])
        if pid not in pids_k:
            continue
        s = qemb_of(pid, 1) @ gemb[keep].T
        ref.append(score_group(s, pids_k, pid, "max"))
    a = np.array(ref)
    r = ("参考 官方1图口径(n>=5人群)", "1", "max",
         float(a[:, 0].mean()), float(a[:, 1].mean()), float(a[:, 2].mean()), len(a))
    print(f"\n=== {r[0]} ===\n1图-max        {r[3]:<9.2%} {r[4]:<9.2%} {r[5]:<9.2%} {r[6]}")
    rows.append(r)

    with open(CSV_OUT, "w", encoding="utf-8-sig") as f:
        f.write("table,k图,fusion,hit@1,hit@3,mAP,n\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]}\n")
    print(f"\n=> CSV 已写 {CSV_OUT}（总耗时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
