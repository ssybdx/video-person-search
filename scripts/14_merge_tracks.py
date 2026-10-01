# -*- coding: utf-8 -*-
"""14 · 轨迹后合并（Track Merging）：用 ReID 特征修复 ByteTrack 的 ID 碎片化

问题：ByteTrack 只靠 IoU（框的运动重叠）关联身份，不看长相。人走近镜头时框从
150px 涨到 750px，IoU 剧变 → 断轨；超过 track_buffer(30帧=1.2s) 未匹配 → 发新 ID。
实测 A3 415 帧发了近 900 个号，同一个人被拆成 10+ 段。

解法：用自训 OSNet 特征做后处理合并。判据（缺一不可）：
  ① 特征相似度 ≥ 分档阈值（按框高：<200px 用 0.80、<300px 用 0.85、≥300px 用 0.90）
     ——小框裁剪图被放大后特征偏糊，同人相似度天然偏低，故阈值随框高降低；
       校准依据：A1 真实 5 人（用户肉眼确认），0.80 档正好还原为 5 组
  ② 时段不重叠（含 GAP_TOL 容差），且合并后组内所有成员两两不重叠
     ——同一人不可能同时出现在两个位置；仅查两两不够，并查集传递合并会把
       A-B、B-C 各自不重叠但 A-C 重叠的三人串成一组（A2 实测 ID29↔ID55 重叠
       3.28s 却因 ID10 传递被误并），故 DSU.can_merge 做组间全量检查
  ③ 框高比 > SCALE_RATIO(1.5) 时，额外要求 mean 相似度 ≥ MEAN_FLOOR(0.72)
     ——max 只取最像的一对帧，易被校服/同款背包等偶然相似欺骗；尺度悬殊
       （远景↔近景）时视角与分辨率都变，max 更不可信，故要求整体都像
     校准依据（30 对人工判定：29 正确 + 1 错误）：错误对 A3 ID2+ID596
       max=0.852 mean=0.707 框高比 2.01 → 被拦；正确对中框高比>1.5 者
       mean 最低 0.732 > 0.72 → 零误伤。单用 max 误伤 6 对、单用 mean 误伤 1 对
判据②是关键防线：实测 A3 的 ID13↔ID218 相似度高达 0.940 但时段重叠，必为两人。

合并后按时间等分重挑 FRAMES_PER_ID 张，避免某人向量组膨胀到几十张。

用法：python scripts/14_merge_tracks.py [A4]     # 不带参数=处理全部已建档案
产物：各视频目录下 index_merged.npz + merged_meta.json（原档案保留不动）
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
INDEX_ROOT = os.path.join(ROOT, "data", "index_real")

MERGE_THRESH = 0.90      # 大框(≥300px)阈值；小框按分档降低，见 thresh_by_box
GAP_TOL = 1.5            # 时段容差(秒)：断档在 1.5s 内仍算不重叠
FRAMES_PER_ID = 8        # 合并后每人重挑几张留底
SCALE_RATIO = 1.5        # 框高比超此值视为"尺度悬殊"（远景↔近景）
MEAN_FLOOR = 0.72        # 尺度悬殊时额外要求的整体相似度下限，见 scale_ok

# ---------- 人工标注约束（优先级高于一切自动判据） ----------
# 背景（2026-10-01）：本批素材为统一校服的小学生，ReID 首要判据（衣着）退化为常量。
# 实测：异人 ID2+ID366 sim=0.805 > 同人 ID1+ID771 sim=0.574——相似度在本素材上
# 不可排序，任何阈值都无法分开二者。故身份判据改以人工标注为最高优先级：
#   MUST_LINK   用户确认同一人 → 强制合并（即使特征不像，如跨尺度漂移）
#   MUST_NOT    用户确认不同人 → 强制拆分（即使特征很像，如同校服同学）
# 键为 (视频标签, 原始ID, 原始ID)，原始ID 以当次运行 13_real_index 的跟踪编号为准。
HUMAN_LINK = {("A3", 1, 771),       # 用户：ID1 与 ID771 是同一个人
              ("A3", 1, 541),       # 用户：ID541 也是目标（遮挡后换号，三段接力）
              ("A2", 29, 10),       # 共现13帧 IoU=0.62 → 同人双号（重复检测发号）
              ("A2", 29, 55)}       # 共现43帧 IoU=0.61 → 同上，四条轨迹全归一人
                                    # 判据补充：时段重叠≠必两人——同框 IoU 高=同一人
                                    # 重复发号；IoU 低=并排两人。共现 IoU 是第三种裁决轴
HUMAN_UNLINK = {("A3", 2, 366)}     # 用户：ID2 与 ID366 不是同一个人


def scale_ok(bh_a, bh_b, sim_mean):
    """尺度悬殊时，要求"整体都像"而非"有一帧很像"。

    max 融合只取最相似的一对帧，易被偶然相似欺骗（校服、同款背包）；mean 反映
    全部帧的整体相似度。尺度差越大（远景↔近景，视角与分辨率都变），max 越不可
    信，故框高比 > SCALE_RATIO 时额外要求 sim_mean ≥ MEAN_FLOOR。

    校准依据（30 对人工判定：29 对正确 + 1 对错误）：
      错误对 A3 ID2+ID596  max=0.852 mean=0.707 框高比 2.01 → 被本规则拦下
      正确对中框高比>1.5 者 mean 最低为 0.732（A2 ID22+ID51）> 0.72 → 零误伤
      A1 ID68+ID150 mean 仅 0.665 但框高比 1.06（尺度一致）→ 不受本规则约束
    单用 max 或单用 mean 均无法零误伤分开（max 误伤 6 对、mean 误伤 1 对）。
    """
    ratio = max(bh_a, bh_b) / max(1e-6, min(bh_a, bh_b))
    if ratio > SCALE_RATIO and sim_mean < MEAN_FLOOR:
        return False, ratio
    return True, ratio


def thresh_by_box(bh_a, bh_b):
    """阈值由较糊的一方（较小框高）决定。

    依据：OSNet 输入 256px，Market 训练裁剪图固定 128px——框高 <200px 时裁剪图
    需被放大，细节丢失导致同人相似度天然偏低（A1 实测：框高 151~226px，同人碎片
    相似度仅 0.808~0.891，用 0.90 阈值一条都合并不上）。A1 真实 5 人由用户肉眼
    确认，用 0.80 档正好得到 5 组，是本阈值的校准依据。
    """
    h = min(bh_a, bh_b)
    if h < 200:
        return 0.80
    if h < 300:
        return 0.85
    return MERGE_THRESH


def imwrite_cn(path, img):
    """中文路径安全写图：cv2.imwrite 遇中文路径会静默失败（坑 1.17），改走字节流。"""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def imread_cn(path):
    """中文路径安全读图：cv2.imread 遇中文路径返回 None（坑 1.17），改走字节流解码。"""
    if not os.path.exists(path):
        return None
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------- 并查集 + 组间一致性约束 ----------
class DSU:
    """带"组间全量一致性"检查的并查集。

    普通并查集只做两两传递：A-B 满足条件、B-C 满足条件 → 三者同组，但 A-C 可能
    不满足。两类实测漏洞：
      ① 时段：A2 的 ID29↔ID55 重叠 3.28s（必为两人），却因都与 ID10 不重叠而被串成一组
      ② 相似度：A3 的 ID366+ID596 sim=0.791<0.80、ID50+ID556 sim=0.786<0.85，
         用户肉眼核对确认"不是一个人"，同样由传递合并造成
    故 can_merge 在合并前检查两组之间**所有跨组成员对**：时段不重叠 AND 相似度达到
    该对的分档阈值，任一不满足即拒绝整次合并。
    """

    def __init__(self, items, sim_map, mean_map, bh_map, must_not=None):
        self.p = {i: i for i in items}
        self.members = {i: {i} for i in items}
        self.sim_map = sim_map      # {frozenset({a,b}): max 相似度}
        self.mean_map = mean_map    # {frozenset({a,b}): mean 相似度}
        self.bh_map = bh_map        # {a: 平均框高}
        self.must_not = must_not or set()   # 人工确认的不同人对（frozenset 键）

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def can_merge(self, a, b, t0, t1):
        """返回 (是否可合并, 拒绝原因)。拒绝原因带类型标记便于日志区分。"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True, None
        for x in self.members[ra]:
            for y in self.members[rb]:
                if frozenset({x, y}) in self.must_not:
                    return False, ("human", x, y)   # 人工确认不同人，禁止同组
                gap = max(t0[x], t0[y]) - min(t1[x], t1[y])
                if gap < -GAP_TOL:          # 两组中存在同时在场的成员 → 必为两人
                    return False, ("overlap", x, y, round(gap, 2))
                key = frozenset({x, y})
                sim = self.sim_map[key]
                th = thresh_by_box(self.bh_map[x], self.bh_map[y])
                if sim < th:                # 组内出现不够像的成员对 → 传递合并过度
                    return False, ("sim", x, y, round(sim, 3), th)
                ok, ratio = scale_ok(self.bh_map[x], self.bh_map[y], self.mean_map[key])
                if not ok:                  # 尺度悬殊且整体不够像 → max 不可信
                    return False, ("scale", x, y, round(ratio, 2),
                                   round(self.mean_map[key], 3), MEAN_FLOOR)
        return True, None

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.p[rb] = ra
        self.members[ra] |= self.members[rb]
        self.members.pop(rb, None)


def load_archive(tag):
    d = os.path.join(INDEX_ROOT, tag)
    npz = os.path.join(d, "index.npz")
    mp = os.path.join(d, "meta.json")
    if not (os.path.exists(npz) and os.path.exists(mp)):
        return None, None, d
    z = np.load(npz)
    meta = json.load(open(mp, encoding="utf-8"))
    db = {}
    for k in z.files:
        if not k.startswith("emb_"):
            continue
        tid = int(k[4:])
        m = meta["ids"][str(tid)]
        db[tid] = {
            "emb": z[k],
            "frames": z[f"frames_{tid}"],
            "paths": [str(p) for p in z[f"paths_{tid}"]],
            "t0": m["first_sec"], "t1": m["last_sec"],
            "n_hits": m["n_hits"], "bh": m["mean_box_h"],
        }
    return db, meta, d


def pairwise(db, tids):
    """两两相似度(max 与 mean) + 时段是否重叠 + 该对适用的分档阈值。

    max：最相似的一对帧，用于主判据（召回优先，任一帧像即算像）；
    mean：全部帧的平均，用于尺度悬殊时的二次判据（见 scale_ok）——尺度差大时
         max 易被校服/同款背包等偶然相似欺骗，mean 要求整体都像才可信。
    """
    out = []
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a, b = tids[i], tids[j]
            s_mat = db[a]["emb"] @ db[b]["emb"].T
            sim = float(s_mat.max())
            sim_mean = float(s_mat.mean())
            gap = max(db[a]["t0"], db[b]["t0"]) - min(db[a]["t1"], db[b]["t1"])
            overlap = gap < -GAP_TOL            # 负得越多重叠越严重
            out.append((a, b, sim, sim_mean, overlap, gap,
                        thresh_by_box(db[a]["bh"], db[b]["bh"])))
    return out


def pick_time_spread(items, n=FRAMES_PER_ID):
    """items: [(frame, path, emb)] 按帧号有序 → 时间等分，每段取一张。"""
    if len(items) <= n:
        return list(items)
    seg = len(items) / n
    picked = []
    for i in range(n):
        chunk = items[int(i * seg):int((i + 1) * seg)]
        if chunk:
            picked.append(chunk[len(chunk) // 2])   # 取段中位，时间上更均匀
    return picked


def merge_one(tag, verbose=True):
    db, meta, d = load_archive(tag)
    if db is None:
        print(f"[跳过] {tag}：未找到 index.npz / meta.json")
        return None
    tids = sorted(db)
    fps = meta["fps"]

    if len(tids) < 2:
        if verbose:
            print(f"\n=== {tag} ===\n  只有 1 个 ID，无需合并")
        return d

    pairs = pairwise(db, tids)
    # 相似度与框高查表，供 can_merge 做组间全量校验（键用 frozenset 保证无向）
    sim_map = {frozenset({a, b}): sim
               for a, b, sim, _sm, _ov, _g, _th in pairs}
    mean_map = {frozenset({a, b}): sm
                for a, b, _sim, sm, _ov, _g, _th in pairs}
    bh_map = {t: db[t]["bh"] for t in tids}
    # 人工约束只取本视频且成员都在当前档案内的条目
    must_link = {frozenset({a, b}) for (tg, a, b) in HUMAN_LINK
                 if tg == tag and a in db and b in db}
    must_not = {frozenset({a, b}) for (tg, a, b) in HUMAN_UNLINK
                if tg == tag and a in db and b in db}
    dsu = DSU(tids, sim_map, mean_map, bh_map, must_not)
    t0map = {t: db[t]["t0"] for t in tids}
    t1map = {t: db[t]["t1"] for t in tids}
    merged_edges, blocked, gb_overlap, gb_sim, gb_scale = [], [], [], [], []
    human_edges = []
    # ① 人工 MUST_LINK 最先落：强制合并，绕过全部自动判据（特征在本素材不可信）
    for pair in sorted(must_link, key=lambda s: min(s)):
        a, b = sorted(pair)
        gap = max(t0map[a], t0map[b]) - min(t1map[a], t1map[b])
        if gap < -GAP_TOL:
            print(f"    ⚠ 人工约束 {a}+{b} 时段重叠{abs(gap):.2f}s——非矛盾：共现 IoU 高"
                  f"证明是跟踪器对同一人重复发号（双号同框），按人工合并")
        dsu.union(a, b)
        human_edges.append((a, b, sim_map[pair], gap))
    # ② 相似度降序贪心：强证据先合并，避免弱边先占位导致组间约束误拒
    for a, b, sim, sim_mean, overlap, gap, th in sorted(pairs, key=lambda x: -x[2]):
        if frozenset({a, b}) in must_not:
            blocked.append((a, b, sim, gap, "human"))   # 人工确认不同人 → 绝对禁止
            continue
        if sim < th:
            continue
        if overlap:
            blocked.append((a, b, sim, gap, "overlap"))  # 长得像但同时在画面 → 两人
            continue
        ok, ratio = scale_ok(db[a]["bh"], db[b]["bh"], sim_mean)
        if not ok:
            # 两两层面就不满足尺度判据（无需进组间检查）：尺度悬殊且整体不够像
            gb_scale.append((a, b, ratio, sim, sim_mean))
            continue
        ok, why = dsu.can_merge(a, b, t0map, t1map)
        if not ok:
            if why[0] == "human":
                gb_overlap.append((a, b, why[1], why[2], sim, "human"))
            elif why[0] == "overlap":
                gb_overlap.append((a, b, why[1], why[2], sim, why[3]))
            elif why[0] == "sim":
                gb_sim.append((a, b, why[1], why[2], sim, why[3], why[4]))
            else:                                  # "scale"：组间传递后尺度悬殊
                gb_scale.append((why[1], why[2], why[3], sim, why[4]))
            continue
        dsu.union(a, b)
        merged_edges.append((a, b, sim, sim_mean, gap, th))

    groups = {}
    for t in tids:
        groups.setdefault(dsu.find(t), []).append(t)
    groups = sorted(groups.values(), key=lambda g: min(g))

    if verbose:
        print(f"\n=== {tag} ===")
        print(f"  原始 {len(tids)} 个 ID → 合并后 {len(groups)} 人"
              f"（分档阈值 {MERGE_THRESH}/{0.85}/{0.80} 按框高，时段容差 {GAP_TOL}s，"
              f"尺度悬殊比>{SCALE_RATIO} 时要求 mean≥{MEAN_FLOOR}）")
        for a, b, sim, sim_mean, gap, th in merged_edges[:10]:
            print(f"    合并 ID{a}+ID{b}  sim={sim:.3f}(阈{th}) mean={sim_mean:.3f} 断档{gap:.2f}s")
        if len(merged_edges) > 10:
            print(f"    …共 {len(merged_edges)} 条合并边")
        if human_edges:
            for a, b, sim, gap in human_edges:
                print(f"    人工约束合并 ID{a}+ID{b}（sim 仅 {sim:.3f}，特征在此素材不可信，"
                      f"以人工标注为准）")
        for a, b, sim, gap, reason in blocked[:6]:
            if reason == "human":
                print(f"    ✗ 拒绝 ID{a}+ID{b}  sim={sim:.3f} → 人工标注为不同人")
            else:
                print(f"    ✗ 拒绝 ID{a}+ID{b}  sim={sim:.3f} 但两人时段重叠{abs(gap):.2f}s")
        for a, b, x, y, sim, g2 in gb_overlap[:5]:
            if g2 == "human":
                print(f"    ✗ 拒绝 ID{a}+ID{b}  sim={sim:.3f}：并入会使组内含人工判异的 "
                      f"ID{x}+ID{y} → 人工约束传递冲突")
            else:
                print(f"    ✗ 拒绝 ID{a}+ID{b}  sim={sim:.3f}：并入会使组内含同时在场的 "
                      f"ID{x}+ID{y}(重叠{abs(g2):.2f}s) → 时段传递冲突")
        for a, b, x, y, sim, sm, th in gb_sim[:5]:
            print(f"    ✗ 拒绝 ID{a}+ID{b}  sim={sim:.3f}：并入会使组内含不够像的 "
                  f"ID{x}+ID{y}(sim={sm:.3f}<阈{th}) → 相似度传递冲突")
        for a, b, ratio, sim, sm in gb_scale[:5]:
            print(f"    ✗ 拒绝 ID{a}+ID{b}  框高比{ratio:.2f}>{SCALE_RATIO} 且 mean={sm:.3f}<{MEAN_FLOOR}"
                  f"（max={sim:.3f} 被偶然相似抬高）→ 尺度悬殊不可信")

    # 组装合并后的档案
    # 注：组内一致性（时段不重叠 AND 相似度达标）已由 DSU.can_merge 在合并前对
    # 全部跨组成员对校验，故此处无需再做组内自检
    store, new_meta = {}, {}
    for gi, g in enumerate(groups, start=1):
        items = []
        for tid in g:
            e, fr, ps = db[tid]["emb"], db[tid]["frames"], db[tid]["paths"]
            for k in range(len(fr)):
                items.append((int(fr[k]), ps[k], e[k]))
        items.sort(key=lambda x: x[0])
        sel = pick_time_spread(items)
        new_tid = gi * 100 + min(g)            # 新 ID：组序号*100 + 最小原 ID，可溯源
        store[f"emb_{new_tid}"] = np.stack([s[2] for s in sel]).astype(np.float32)
        store[f"frames_{new_tid}"] = np.array([s[0] for s in sel], dtype=np.int32)
        store[f"paths_{new_tid}"] = np.array([s[1] for s in sel])
        allf = [it[0] for it in items]
        new_meta[str(new_tid)] = {
            "merged_from": g,
            # 原始 crops 目录名（合并编号与磁盘文件夹是两套命名空间，此处给映射）
            "crops_dirs": [f"ID{t:03d}" for t in sorted(g)],
            "n_crops_total": len(items), "n_kept": len(sel),
            "first_frame": min(allf), "last_frame": max(allf),
            "first_sec": round(min(allf) / fps, 2),
            "last_sec": round(max(allf) / fps, 2),
            "n_track_hits": sum(db[t]["n_hits"] for t in g),
            "mean_box_h": round(float(np.mean([db[t]["bh"] for t in g])), 1),
        }

    np.savez(os.path.join(d, "index_merged.npz"), **store)

    # 生成人工核对文件夹：每组合并一张拼图，各原始 ID 取一张代表帧并标注来源目录，
    # 让人一眼看出"这几个碎片是不是同一个人"（crops 用原始 ID 命名，合并编号在磁盘上
    # 无对应目录，此前用户按合并编号找不到路径）
    review_dir = os.path.join(d, "review")
    os.makedirs(review_dir, exist_ok=True)
    n_sheet = 0
    for tid, m in new_meta.items():
        if len(m["merged_from"]) < 2:
            continue                                  # 单条轨迹无需核对
        tiles = []
        for t in sorted(m["merged_from"]):
            ps = db[t]["paths"]
            img = imread_cn(ps[len(ps) // 2]) if len(ps) else None
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = 260.0 / h if h > 260 else 1.0
            img = cv2.resize(img, (int(w * scale), int(img.shape[0] * scale)))
            cv2.putText(img, f"ID{t:03d}", (4, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(img)
        if len(tiles) < 2:
            continue
        th = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
                                    cv2.BORDER_CONSTANT, value=(30, 30, 30))
                 for t in tiles]
        sheet = cv2.hconcat(tiles)
        if imwrite_cn(os.path.join(review_dir, f"merged_{int(tid)}.jpg"), sheet):
            n_sheet += 1
    out_meta = dict(meta)
    out_meta["merge"] = {"thresh_large": MERGE_THRESH, "gap_tol": GAP_TOL,
                         "frames_per_id": FRAMES_PER_ID,
                         "n_before": len(tids), "n_after": len(groups),
                         "n_edges": len(merged_edges), "n_blocked": len(blocked),
                         "n_reject_overlap": len(gb_overlap),
                         "n_reject_sim": len(gb_sim),
                         "n_reject_scale": len(gb_scale),
                         "human_link": [sorted(p) for p in must_link],
                         "human_unlink": [sorted(p) for p in must_not],
                         "n_human_edges": len(human_edges)}
    out_meta["ids"] = new_meta
    with open(os.path.join(d, "merged_meta.json"), "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  合并后档案（ID = 合并后新编号；crops 目录仍用原始 ID 命名）：")
        for tid, m in new_meta.items():
            tag = "" if len(m["merged_from"]) < 2 else "  ← 建议核对"
            print(f"    ID{int(tid):<5} ← crops/{', crops/'.join(m['crops_dirs'])}"
                  f"  留底{m['n_kept']}张  时段 {m['first_sec']}~{m['last_sec']}s"
                  f"  框高{m['mean_box_h']}px{tag}")
        print(f"  统计：合并 {len(merged_edges)} 边 | 两两拒绝(重叠) {len(blocked)} | "
              f"组间拒绝(时段冲突) {len(gb_overlap)} | 组间拒绝(相似度冲突) {len(gb_sim)} | "
              f"拒绝(尺度悬殊) {len(gb_scale)}")
        print(f"  => index_merged.npz + merged_meta.json 已存 {d}")
        print(f"  => 人工核对拼图 {n_sheet} 张：{os.path.join(d, 'review')}"
              f"（每张横向拼接同一组的各原始 ID 代表帧，标注来源目录）")
    return d


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    tags = ([arg] if arg else
            sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(INDEX_ROOT, "*", "index.npz"))))
    if not tags:
        print(f"未找到已建档案：{INDEX_ROOT}\\*\\index.npz（先跑 13_real_index.py）")
        return
    print(f"待合并 {len(tags)} 个档案库：{tags}")
    for t in tags:
        merge_one(t)
    print("\n全部完成。查询端将自动优先使用 index_merged.npz。")


if __name__ == "__main__":
    main()
