# -*- coding: utf-8 -*-
"""18 · 端到端演示：照片进 -> 轨迹出（项目最初定义的那句话）

流程：查询照片(data/query/A) -> YOLO 人体裁剪 -> A2 权重提特征
     -> 跨四段合并档案 max 融合检索 -> 每段输出 Top-3 + 命中组时段/代表截图
判卷逻辑全部复用 16 号已验证实现（query_feat / build_gallery / search），不另写。

产物：data\demo\result.json（结构化结果）
     data\demo\evidence\<seg>_G<gid>.jpg（各命中组的留底拼图，供报告/GIF 用）
用法：python scripts/18_end_to_end_demo.py
"""
import glob
import importlib
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
m16 = importlib.import_module("16_b_group_ablation")   # 复用查询/检索实现

GT_PATH = os.path.join(ROOT, "data", "index_real", "ground_truth.json")
DEMO_DIR = os.path.join(ROOT, "data", "demo")
EVID_DIR = os.path.join(DEMO_DIR, "evidence")


def imread_cn(path):
    if not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def make_evidence(seg, gid, paths):
    """把命中组的全部留底 crop 拼成一张证据图（标注文件名），存 evidence/"""
    tiles = []
    for p in paths:
        img = imread_cn(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        sc = 240.0 / h
        img = cv2.resize(img, (int(w * sc), 240))
        name = os.path.basename(p)
        cv2.rectangle(img, (0, 0), (img.shape[1], 20), (40, 40, 40), -1)
        cv2.putText(img, name, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1)
        tiles.append(img)
    if not tiles:
        return None
    ncol = min(4, len(tiles))
    rows = [tiles[i:i + ncol] for i in range(0, len(tiles), ncol)]
    # 各 tile 宽度不一（crop 长宽比不同）：统一 pad/裁到同宽再横拼、纵拼
    W = max(t.shape[1] for t in tiles)
    def fitw(t):
        if t.shape[1] == W:
            return t
        if t.shape[1] > W:
            return t[:, :W]
        return cv2.copyMakeBorder(t, 0, 0, 0, W - t.shape[1],
                                  cv2.BORDER_CONSTANT, value=(30, 30, 30))
    rows = [[fitw(x) for x in r] + [np.zeros((240, W, 3), np.uint8)] * (ncol - len(r))
            for r in rows]
    sheet = np.vstack([np.hstack(r) for r in rows])
    out = os.path.join(EVID_DIR, f"{seg}_G{gid}.jpg")
    ok, buf = cv2.imencode(".jpg", sheet)
    if ok:
        buf.tofile(out)
    return out


def main():
    gt = json.load(open(GT_PATH, encoding="utf-8"))
    model = m16.load_reid()
    yolo = YOLO(m16.YOLO_W)
    photos = sorted(glob.glob(os.path.join(m16.QUERY_DIR, "*.jpg")))
    print(f"查询照片 {len(photos)} 张（目标人物见 ground_truth.json）")
    qall = np.stack([m16.query_feat(model, yolo, p) for p in photos])
    gal = m16.build_gallery()
    ranked = m16.search(qall, gal, "max")               # 多图 max 融合 = 系统主策略

    os.makedirs(EVID_DIR, exist_ok=True)
    result = {"query_photos": [os.path.basename(p) for p in photos],
              "fusion": "max(4图×档案向量)", "videos": {}}
    print("\n================ 端到端检索结果 ================")
    for seg in m16.SEGMENTS:
        top = [(g, v) for s, g, v in ranked if s == seg][:3]
        mm = json.load(open(os.path.join(ROOT, "data", "index_real", seg,
                                         "merged_meta.json"), encoding="utf-8"))
        z = np.load(os.path.join(ROOT, "data", "index_real", seg, "index_merged.npz"))
        tgt_gid = gt[seg]["target_group"]
        entry = {"top3": [], "target_rank": None, "target_time_sec": None,
                 "evidence": None}
        print(f"\n--- {seg}（档案 {mm['merge']['n_after']} 人）---")
        for rank, (g, v) in enumerate(top, start=1):
            hit = " <= 目标人物" if g == tgt_gid else ""
            print(f"  #{rank} G{g}  sim={v:.4f}{hit}")
            entry["top3"].append({"group": g, "sim": round(v, 4),
                                  "is_target": g == tgt_gid})
        all_seg = [(g, v) for s, g, v in ranked if s == seg]
        ranks = [i + 1 for i, (g, _v) in enumerate(all_seg) if g == tgt_gid]
        if ranks:
            r = ranks[0]
            entry["target_rank"] = r
            tm = mm["ids"][str(tgt_gid)]
            entry["target_time_sec"] = [tm["first_sec"], tm["last_sec"]]
            entry["target_orig_ids"] = tm["merged_from"]
            ev = make_evidence(seg, tgt_gid, [str(p) for p in z[f"paths_{tgt_gid}"]])
            entry["evidence"] = os.path.relpath(ev, ROOT) if ev else None
            print(f"  => 目标 G{tgt_gid} 排名 #{r}/{len(all_seg)}，"
                  f"出现时段 {tm['first_sec']}~{tm['last_sec']}s")
        result["videos"][seg] = entry

    with open(os.path.join(DEMO_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    n_hit = sum(1 for e in result["videos"].values() if e["target_rank"] == 1)
    print(f"\n总结：4 段中 Top-1 命中目标 {n_hit}/4；"
          f"详情 data\\demo\\result.json，证据拼图 data\\demo\\evidence\\")


if __name__ == "__main__":
    main()
