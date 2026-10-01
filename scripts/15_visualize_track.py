# -*- coding: utf-8 -*-
"""15 · 跟踪可视化：把检测框+轨迹ID画回视频，输出带标注的 mp4

作用（三件）：
1. 肉眼验证 ID 碎片化与合并质量——按"原始 ID"着色，同一个人被拆成的多个碎片
   会是不同颜色；若两段颜色接力出现（一个消失、另一个紧接着在附近出现），
   即为同一人断轨，对应 merged_meta.json 里的一个合并组
2. 直观展示 ByteTrack 的断轨现场——track_buffer=30 帧（25fps 下仅 1.2s），
   人走近镜头时框剧变即断轨，是 14 号脚本要解决的问题的可视化证据
3. 演示素材——答辩/简历 GIF 的直接来源

标注内容：框 + 原始ID + 合并组号(G) + 当前框高(h)，左上角 HUD 显示帧号/时刻/在场人数。
框高实时显示是为便于判断"尺度悬殊"（14-5 的判据依据：ID002 h196 vs ID596 h393）。

实现要点：
- cv2.VideoWriter 遇中文路径静默失败（同坑 1.17），故先写 ASCII 临时路径再移动
- cv2.putText 无法渲染中文，标注一律 ASCII
- 按原始 ID 着色（而非合并组），碎片才看得出来；组号以文字标注

用法：python scripts/15_visualize_track.py [A4]      # 不带参数=处理全部 A*.mp4
产物：data/vis/A1_vis.mp4 等
"""
import colorsys
import glob
import json
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
VIDEO_DIR = os.path.join(ROOT, "data", "videos")
INDEX_ROOT = os.path.join(ROOT, "data", "index_real")
OUT_DIR = os.path.join(ROOT, "data", "vis")
YOLO_W = os.path.join(ROOT, "models", "yolov8n.pt")

CONF = 0.2          # 与 13_real_index.py 一致，否则 ID 编号对不上
OUT_W = 1280        # 输出宽度（1440p 原图缩到 720p，文件小、播放流畅，人脸衣着仍清晰）
MIN_BOX_H = 40      # 输出图上低于此高度的框不画（远处噪点，避免满屏小框）


def color_for(idx):
    """按序号取一个视觉区分度高的颜色（黄金角均匀分布色相）。"""
    hue = (idx * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))      # BGR


def load_group_map(tag):
    """原始 ID -> 合并组号；无 merged_meta.json 时返回空表（不影响出图）。"""
    p = os.path.join(INDEX_ROOT, tag, "merged_meta.json")
    if not os.path.exists(p):
        return {}
    mm = json.load(open(p, encoding="utf-8"))
    out = {}
    for new_id, m in mm.get("ids", {}).items():
        for orig in m.get("merged_from", []):
            out[orig] = int(new_id)
    return out


def annotate(video_path, tag, verbose=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    group_map = load_group_map(tag)

    det = YOLO(YOLO_W)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[FAIL] 无法解码 {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = OUT_W / src_w
    out_h = int(round(src_h * scale))

    # 先写 ASCII 临时路径（VideoWriter 中文路径会静默产出 0 字节文件）
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix=f"vis_{tag}_")
    os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (OUT_W, out_h))
    if not writer.isOpened():
        cap.release()
        os.remove(tmp_path)
        print(f"[FAIL] VideoWriter 打不开（编解码器缺失？）")
        return None

    colors, idx = {}, 0
    n_box_total, ids_seen = 0, set()
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = det.track(frame, persist=True, tracker="bytetrack.yaml",
                        classes=[0], conf=CONF, verbose=False)
        boxes = res[0].boxes
        active = 0
        if boxes is not None and boxes.id is not None:
            xyxy, ids, confs = boxes.xyxy, boxes.id, boxes.conf
            for b, tid, c in zip(xyxy, ids, confs):
                tid = int(tid)
                x1, y1, x2, y2 = [int(v) for v in b.cpu().numpy()]
                if (y2 - y1) < MIN_BOX_H:
                    continue
                if tid not in colors:
                    colors[tid] = color_for(len(colors))
                    ids_seen.add(tid)
                col = colors[tid]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
                # 标签底板：ID + 合并组 + 框高 + 置信度
                g = group_map.get(tid)
                label = f"ID{tid}" + (f" G{g}" if g is not None else "") \
                        + f" h{y2-y1} {float(c):.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                ty = max(y1 - th - 8, 0)
                cv2.rectangle(frame, (x1, ty), (x1 + tw + 6, ty + th + 8), col, -1)
                cv2.putText(frame, label, (x1 + 3, ty + th + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
                active += 1
                n_box_total += 1

        # HUD（左上角）
        hud = (f"{tag}  frame {idx}/{n_frames}  t={idx/fps:.2f}s  "
               f"active={active}  ids_total={len(ids_seen)}")
        cv2.rectangle(frame, (0, 0), (620, 34), (20, 20, 20), -1)
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2)

        out = cv2.resize(frame, (OUT_W, out_h), interpolation=cv2.INTER_AREA)
        writer.write(out)
        idx += 1
        del frame, out

    cap.release()
    writer.release()

    final = os.path.join(OUT_DIR, f"{tag}_vis.mp4")
    shutil.move(tmp_path, final)          # 用 move 落到中文路径，绕开 VideoWriter 限制
    size_mb = os.path.getsize(final) / 1024 / 1024

    if verbose:
        print(f"\n=== {tag} ===")
        print(f"  源 {src_w}x{src_h} @{fps:.1f}fps / {n_frames}帧 "
              f"-> 输出 {OUT_W}x{out_h}")
        print(f"  处理 {idx} 帧，画出 {n_box_total} 个框，共见 {len(ids_seen)} 个原始ID"
              f"，耗时 {time.time()-t0:.0f}s")
        if group_map:
            groups = sorted(set(group_map.values()))
            multi = [g for g in groups
                     if sum(1 for v in group_map.values() if v == g) > 1]
            print(f"  合并组 {len(groups)} 个（其中 {len(multi)} 个由多碎片合并）")
        print(f"  => {final}  ({size_mb:.1f} MB)")
    return final


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg:
        tag = arg.replace(".mp4", "")
        vids = [os.path.join(VIDEO_DIR, f"{tag}.mp4")]
    else:
        vids = sorted(glob.glob(os.path.join(VIDEO_DIR, "A*.mp4")))
    vids = [v for v in vids if os.path.exists(v)]
    if not vids:
        print(f"未找到视频：{VIDEO_DIR}\\A*.mp4")
        return
    print(f"待渲染 {len(vids)} 段：{[os.path.basename(v) for v in vids]}")
    for v in vids:
        annotate(v, os.path.splitext(os.path.basename(v))[0])
    print(f"\n全部完成，输出在 {OUT_DIR}")


if __name__ == "__main__":
    main()
