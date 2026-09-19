# -*- coding: utf-8 -*-
"""MOT16 帧序列 -> mp4 视频，供索引端 pipeline 用真实监控场景测试。

用法: python scripts/12_mot16_to_video.py [序列名 默认MOT16-13] [帧数 默认250]
输出: data/videos/<序列名>.mp4
"""
import os
import sys
import glob

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOT16 = os.path.join(ROOT, "MOT16", "train")


def imread_u(path):
    """支持中文路径的 imread：先按二进制读进内存再解码。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def seq_to_video(seq="MOT16-13", n_frames=250, fps=25):
    imgs = sorted(glob.glob(os.path.join(MOT16, seq, "img1", "*.jpg")))
    if not imgs:
        sys.exit(f"序列不存在或无jpg: {seq}")
    imgs = imgs[:n_frames]
    first = imread_u(imgs[0])
    h, w = first.shape[:2]
    out = os.path.join(ROOT, "data", "videos", f"{seq}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in imgs:
        vw.write(imread_u(p))
    vw.release()
    print(f"OK {seq}: {len(imgs)}帧 {w}x{h} -> {out} ({os.path.getsize(out)/1e6:.1f}MB)")
    return out


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "MOT16-13"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    seq_to_video(seq, n)
