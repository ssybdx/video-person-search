# -*- coding: utf-8 -*-
"""13 · 真实素材索引端：4K 多视频流式建库（自训 A2 权重）

与教学版 10_index_pipeline.py 的三点关键区别：
1. 流式两遍扫描——4K 单帧 23.7MB，2441 帧全驻留需 56GB（本机可用仅 7.8GB）必爆；
   改为第一遍逐帧跟踪只记 (帧号,框)、帧用完即释放，第二遍顺序重读按需裁剪
2. 多视频批量——data/videos/VID_*.mp4 每段独立档案库，YOLO 实例每段新建以重置跟踪器
3. 权重——自训 A2（07B-2 选型冠军，余弦协议 Rank-1 91.69%），非 ImageNet/第三方

留底策略：把某 ID 的命中帧按时间等分成 N 段，每段取面积最大的框
（纯面积 top-k 会让留底全挤在"人走近镜头"那几秒，时间等分才能铺开轨迹）

用法：
  python scripts/13_real_index.py                 # 全部 VID_*.mp4
  python scripts/13_real_index.py VID_2026...mp4  # 只建指定一段
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchreid
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

VIDEO_DIR = os.path.join(ROOT, "data", "videos")
OUT_ROOT = os.path.join(ROOT, "data", "index_real")
YOLO_W = os.path.join(ROOT, "models", "yolov8n.pt")
REID_W = os.path.join(ROOT, "models", "osnet_market_A2_20260928.pth")

CONF = 0.2            # 检测置信度下限：ByteTrack 要吃低分框，别用默认 0.25
MIN_FRAMES = 30       # 轨迹至少持续 30 帧（60fps≈0.5s），滤掉一闪而过的噪声
MIN_BOX_H = 150       # 4K 下框高 < 150px（帧高 7%）的留底帧丢弃，特征太糊
FRAMES_PER_ID = 8     # 每个 ID 最多留几张（B 组消融最多用 5 图，8 张有余量）
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_H, IMG_W = 256, 128


# ---------- ReID 模型（自训 A2） ----------
def load_reid(ckpt=REID_W):
    model = torchreid.models.build_model(name="osnet_x1_0", num_classes=751,
                                         pretrained=False)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in raw.items()}   # 剥 DataParallel 前缀
    ret = model.load_state_dict(sd, strict=False)
    assert not ret.missing_keys and not ret.unexpected_keys, \
        f"权重不匹配 missing={ret.missing_keys[:3]} unexpected={ret.unexpected_keys[:3]}"
    print(f"=> ReID 权重已加载: {os.path.basename(ckpt)}（567 键全匹配）")
    return model.eval().cuda()


TF = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def extract_feats(model, paths, batch=64):
    """图片路径列表 -> (N,512) 已 L2 归一化的特征矩阵。"""
    outs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = torch.stack([TF(Image.open(p).convert("RGB"))
                                for p in paths[i:i + batch]])
            outs.append(model(imgs.cuda()))
    return torch.nn.functional.normalize(torch.cat(outs), dim=1)


# ---------- 中文路径安全写图（cv2.imwrite 会静默失败） ----------
def imwrite_cn(path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def clip_box(box, w, h):
    x1, y1, x2, y2 = [int(v) for v in box]
    return max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


# ---------- 第一遍：流式跟踪，只记坐标不留帧 ----------
def track_stream(path):
    """逐帧 YOLO+ByteTrack，返回 ({tid: [(帧号,框,conf)]}, 总帧数, fps)。内存只驻留 1 帧。"""
    det = YOLO(YOLO_W)                 # 每段视频新建实例，跟踪器状态干净、ID 不串号
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法解码视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    tracks, idx = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = det.track(frame, persist=True, tracker="bytetrack.yaml",
                        classes=[0], conf=CONF, verbose=False)
        boxes = res[0].boxes
        if boxes.id is not None:
            xyxy, confs, ids = boxes.xyxy, boxes.conf, boxes.id
            for tid, b, c in zip(ids, xyxy, confs):
                tracks.setdefault(int(tid), []).append(
                    (idx, b.cpu().numpy(), float(c)))
        idx += 1
        del frame                      # 显式释放，4K 单帧 23.7MB
    cap.release()
    return tracks, idx, fps


# ---------- 留底帧挑选：时间等分 + 每段取最大框 ----------
def pick_frames(hits, n=FRAMES_PER_ID):
    """hits 已按帧号有序。等分时间轴取 n 段，每段挑面积最大者——轨迹铺开且都清晰。"""
    if len(hits) <= n:
        return list(hits)
    seg = len(hits) / n
    picked = []
    for i in range(n):
        chunk = hits[int(i * seg):int((i + 1) * seg)]
        if chunk:
            picked.append(max(chunk, key=lambda h: box_area(h[1])))
    return picked


# ---------- 第二遍：顺序重读，命中帧号即裁剪 ----------
def crop_pass(path, picks, out_dir):
    """picks: {tid: [(帧号,框,conf)]}。返回 [(crop路径, tid, 帧号)]。"""
    need = {}
    for tid, lst in picks.items():
        for fi, b, c in lst:
            need.setdefault(fi, []).append((tid, b, c))
    crops_dir = os.path.join(out_dir, "crops")
    cap = cv2.VideoCapture(path)
    saved, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in need:
            h, w = frame.shape[:2]
            for tid, b, c in need[idx]:
                x1, y1, x2, y2 = clip_box(b, w, h)
                if (y2 - y1) < MIN_BOX_H:          # 太小/太糊，不留
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                d = os.path.join(crops_dir, f"ID{tid:03d}")
                os.makedirs(d, exist_ok=True)
                p = os.path.join(d, f"f{idx:05d}.jpg")
                if imwrite_cn(p, crop):
                    saved.append((p, tid, idx))
        idx += 1
        del frame
    cap.release()
    return saved


# ---------- 单段视频：建库 ----------
def build_one(video_path, reid, verbose=True):
    tag = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    tracks, n_frames, fps = track_stream(video_path)
    t_track = time.time() - t0
    if verbose:
        print(f"\n=== {tag} ===")
        print(f"  总帧数 {n_frames} / fps {fps:.1f} / 时长 {n_frames/fps:.1f}s "
              f"| 第一遍跟踪 {t_track:.0f}s | 原始轨迹 {len(tracks)} 条")

    # 过滤噪声轨迹（持续太短）
    valid = {tid: h for tid, h in tracks.items() if len(h) >= MIN_FRAMES}
    if verbose:
        print(f"  持续≥{MIN_FRAMES}帧的轨迹: {len(valid)} 条"
              f"（滤掉 {len(tracks)-len(valid)} 条一闪而过的噪声）")

    picks = {tid: pick_frames(sorted(h, key=lambda x: x[0]))
             for tid, h in valid.items()}
    t0 = time.time()
    saved = crop_pass(video_path, picks, out_dir)
    if verbose:
        print(f"  第二遍裁剪 {time.time()-t0:.0f}s -> 留底 {len(saved)} 张"
              f"（框高<{MIN_BOX_H}px 的已丢弃）")

    if not saved:
        print("  !! 无有效留底，跳过建库")
        return None

    # 按 tid 分组提特征
    paths = [s[0] for s in saved]
    feats = extract_feats(reid, paths).cpu().numpy()
    by_tid = {}
    for i, (p, tid, fi) in enumerate(saved):
        by_tid.setdefault(tid, []).append((fi, p, feats[i]))

    store, meta_ids = {}, {}
    for tid, items in sorted(by_tid.items()):
        items.sort(key=lambda x: x[0])
        store[f"emb_{tid}"] = np.stack([it[2] for it in items]).astype(np.float32)
        store[f"frames_{tid}"] = np.array([it[0] for it in items], dtype=np.int32)
        store[f"paths_{tid}"] = np.array([it[1] for it in items])
        allf = sorted(f for f, _, _ in
                      [(h[0], h[1], h[2]) for h in valid[tid]])
        meta_ids[str(tid)] = {
            "n_hits": len(valid[tid]),
            "n_kept": len(items),
            "first_frame": allf[0], "last_frame": allf[-1],
            "first_sec": round(allf[0] / fps, 2),
            "last_sec": round(allf[-1] / fps, 2),
            "mean_box_h": round(float(np.mean(
                [h[1][3] - h[1][1] for h in valid[tid]])), 1),
        }
    np.savez(os.path.join(out_dir, "index.npz"), **store)
    meta = {"video": os.path.basename(video_path), "fps": round(fps, 2),
            "n_frames": n_frames, "duration_sec": round(n_frames / fps, 2),
            "reid_weight": os.path.basename(REID_W),
            "params": {"conf": CONF, "min_frames": MIN_FRAMES,
                       "min_box_h": MIN_BOX_H, "frames_per_id": FRAMES_PER_ID},
            "ids": meta_ids}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  => 档案库 {len(by_tid)} 人 | npz + meta.json 已存 {out_dir}")
        for tid, m in sorted(meta_ids.items(), key=lambda kv: int(kv[0])):
            print(f"     ID{int(tid):03d}: 命中{m['n_hits']}帧 留底{m['n_kept']}张 "
                  f"时段 {m['first_sec']}~{m['last_sec']}s 平均框高{m['mean_box_h']}px")
    return out_dir


def main():
    argv = sys.argv[1:]
    global MIN_FRAMES
    pos, i = [], 0
    while i < len(argv):                       # 解析 --min-frames N（人工约束可放宽准入）
        if argv[i] == "--min-frames" and i + 1 < len(argv):
            MIN_FRAMES = int(argv[i + 1])
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    arg = pos[0] if pos else None
    vids = ([os.path.join(VIDEO_DIR, arg)] if arg
            else sorted(glob.glob(os.path.join(VIDEO_DIR, "A*.mp4"))))
    if not vids:
        print(f"未找到视频：{VIDEO_DIR}\\A*.mp4")
        return
    print(f"待处理 {len(vids)} 段视频（MIN_FRAMES={MIN_FRAMES}）")
    reid = load_reid()
    done = []
    for v in vids:
        r = build_one(v, reid)
        if r:
            done.append(r)
    print(f"\n全部完成：{len(done)}/{len(vids)} 段建成档案库")


if __name__ == "__main__":
    main()
