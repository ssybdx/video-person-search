# -*- coding: utf-8 -*-
"""
10 · 索引端 pipeline：视频 -> 检测 -> 跟踪 -> 每ID留底 -> OSNet特征 -> 档案库
完整实现版（指导者代写，用户须能复述每个函数职责）。冒烟入口同07：
  python scripts/10_index_pipeline.py 1/2/3  (逐级单测)
  python scripts/10_index_pipeline.py        (全部) -> 生成 data/index/index.npz 档案库
特征端直接复用07的现成函数（同目录 import）。
"""
import os
import importlib

import cv2
import numpy as np
from ultralytics import YOLO

m07 = importlib.import_module("07_market_eval")   # 复用 07：build_reid_model / extract_feats

VIDEO = r"E:\QianwenApp\workplaces\视频目标人物检索项目\data\videos\people.mp4"
WEIGHTS = r"E:\QianwenApp\workplaces\视频目标人物检索项目\models\yolov8n.pt"
OUT_DIR = r"E:\QianwenApp\workplaces\视频目标人物检索项目\data\index"
FRAME_STEP = 2          # 每2帧取1帧：CPU跑得动的折中
CONF = 0.2              # 检测置信度下限：ByteTrack要吃低分框，别用默认0.25以上
FRAMES_PER_ID = 5       # 每个track ID最多留几帧
MIN_HITS = 3            # 命中帧数少于这个的ID视为噪声轨迹


# ---------- 1: 抽帧 ----------
def read_frames(path, step=FRAME_STEP):
    """用cv2按step间隔读帧，返回 [np.ndarray(BGR,H,W,3), ...]。"""
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:                     # 读到底，ok=False
            break
        if i % step == 0:              # 每step帧留一张
            frames.append(frame)
        i += 1
    cap.release()
    return frames


# ---------- 2: 检测+跟踪，逐帧记录命中 ----------
def track_video(detector, frames):
    """逐帧 model.track，汇总 {track_id: [(帧序号, box, conf), ...]}。"""
    tracks = {}
    for idx, frame in enumerate(frames):
        results = detector.track(frame, persist=True, tracker="bytetrack.yaml",
                                 classes=[0], conf=CONF, verbose=False)
        boxes = results[0].boxes
        ids = boxes.id
        if ids is None:                # 这帧一个人没检到（或跟踪器还没发号）
            continue
        xyxy, confs = boxes.xyxy, boxes.conf
        for tid, box, c in zip(ids, xyxy, confs):
            tid = int(tid)
            tracks.setdefault(tid, []).append((idx, box.cpu().numpy(), float(c)))
    return tracks


# ---------- 3: 每ID挑留底帧（面积优先，兼顾时间覆盖） ----------
def pick_frames(hits):
    """从某ID全部命中挑最多FRAMES_PER_ID张：面积大=离镜头近细节全；再按帧号排=时间也铺开。"""
    scored = sorted(hits, key=lambda h: (h[1][2] - h[1][0]) * (h[1][3] - h[1][1]),
                    reverse=True)[:FRAMES_PER_ID]
    return sorted(scored, key=lambda h: h[0])


# ---------- 4: 裁剪+提特征+存库 ----------
def build_index(frames, tracks, reid):
    """留底帧裁剪存jpg -> 复用07提特征 -> np.savez 打包档案库。"""
    crops_dir = os.path.join(OUT_DIR, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    samples, owners = [], []           # (path,tid,0)喂给07的extract_feats；owners记录行归属
    kept = {}
    for tid in sorted(tracks):
        hits = tracks[tid]
        if len(hits) < MIN_HITS:       # 一闪而过的噪声轨迹，不留
            continue
        sel = pick_frames(hits)
        id_dir = os.path.join(crops_dir, f"ID{tid:02d}")
        os.makedirs(id_dir, exist_ok=True)
        h, w = frames[0].shape[:2]
        for fi, (frame_idx, box, _c) in enumerate(sel):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)          # 坐标防越界
            x2, y2 = min(x2, w), min(y2, h)
            crop = frames[frame_idx][y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cp = os.path.join(id_dir, f"c{fi}_f{frame_idx:04d}.jpg")
            cv2.imwrite(cp, crop)
            samples.append((cp, tid, 0))
            owners.append(tid)
        kept[tid] = sel

    feats = m07.extract_feats(reid, samples)          # (N,512) 已L2归一化
    save = {}
    for tid in kept:
        rows = [i for i, o in enumerate(owners) if o == tid]
        save[f"emb_{tid}"] = feats[rows].cpu().numpy()  # 存整个向量组，不平均！max融合要用（.cpu()：07 GPU 化后特征在显存）
        save[f"frames_{tid}"] = np.array([h[0] for h in kept[tid]]) * FRAME_STEP  # 换算回原视频帧号
        save[f"paths_{tid}"] = np.array([samples[i][0] for i in rows])
    np.savez(os.path.join(OUT_DIR, "index.npz"), **save)

    for tid in sorted(kept):
        total = len(tracks[tid])
        area = np.mean([(b[2] - b[0]) * (b[3] - b[1]) for _, b, _ in kept[tid]])
        print(f"  ID {tid:2d}: 命中{total}帧 -> 留底{len(kept[tid])}帧, 平均框面积{area:.0f}px²")
    print(f"档案库已存: {os.path.join(OUT_DIR, 'index.npz')}")


# ---------- 主流程 ----------
def main():
    frames = read_frames(VIDEO)
    print(f"抽帧完成：{len(frames)} 帧")
    detector = YOLO(WEIGHTS)
    tracks = track_video(detector, frames)
    print(f"跟踪完成：{len(tracks)} 个ID")
    reid = m07.build_reid_model()
    build_index(frames, tracks, reid)


if __name__ == "__main__":
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "1":
        fs = read_frames(VIDEO)
        print(len(fs), fs[0].shape)
    elif step == "2":
        fs = read_frames(VIDEO)
        tr = track_video(YOLO(WEIGHTS), fs)
        print({k: len(v) for k, v in sorted(tr.items())})
    elif step == "3":
        fs = read_frames(VIDEO)
        tr = track_video(YOLO(WEIGHTS), fs)
        for k, v in sorted(tr.items()):
            print(k, len(pick_frames(v)))
    else:
        main()
