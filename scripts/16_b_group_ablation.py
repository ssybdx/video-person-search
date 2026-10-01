# -*- coding: utf-8 -*-
"""16 · B 组检索消融：单图/多图 × max/mean 命中率矩阵（跨视频 gallery）

实验设计（training_plan.md 的 B 组，按现有素材缩为四档）：
  B1 单图max   用 query/A 第 1 张照片
  B2 三图max   前 3 张，组内跨查询取 max
  B3 四图max   全部 4 张（多图卖点的完整形态）
  B4 四图mean  查询侧 4 向量先平均再比对（朴素基线，用来证明 max 优于 mean）

流程与口径：
  查询照片 -> YOLO 检人取最高框裁剪 -> A2 权重提特征（与索引端同预处理，同构可比）
  gallery  = 四段 index_merged.npz 的全部组合并（A1×5 + A2×9 + A3×29 + A4×1 = 44 组），
             跨段检索更接近真实系统，负样本也更多
  命中目标 = 目标组出现在排序前列。A2 存在未裁决争议（G101/G310 时段重叠，见
             ground_truth.json conflict 字段），故报双口径：
               strict  = 目标组记 A1:368, A2:101, A3:101, A4:101（共4组）
               lenient = A2 允许 101 或 310 任一算目标（共5组）

输出：终端矩阵 + docs/b_group_results.csv
"""
import glob
import json
import os

import cv2
import numpy as np
import torch
import torchreid
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
IDX_ROOT = os.path.join(ROOT, "data", "index_real")
QUERY_DIR = os.path.join(ROOT, "data", "query", "A")
YOLO_W = os.path.join(ROOT, "models", "yolov8n.pt")
REID_W = os.path.join(ROOT, "models", "osnet_market_A2_20260928.pth")
CSV_OUT = os.path.join(ROOT, "docs", "b_group_results.csv")

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
TF = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

SEGMENTS = ["A1", "A2", "A3", "A4"]
# A2 悬案已于 2026-10-01 裁决（ground_truth.json）：G101×G310 共现 IoU 0.61~0.62
# = 同一人重复发号，人工约束合并后目标 = G310(原10,29,55)。strict/lenient 统一。
TARGETS_STRICT = {"A1": [368], "A2": [310], "A3": [101], "A4": [101]}
TARGETS_LENIENT = {"A1": [368], "A2": [310], "A3": [101], "A4": [101]}


def load_reid():
    model = torchreid.models.build_model(name="osnet_x1_0", num_classes=751,
                                         pretrained=False)
    ck = torch.load(REID_W, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().cuda()


def query_feat(model, yolo, path):
    """照片 -> YOLO 取最高人体框裁剪 -> (512,) L2 归一化特征"""
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None, f"读不了: {path}"
    r = yolo.predict(bgr, classes=0, conf=0.25, verbose=False, device="cuda")[0]
    assert r.boxes is not None and len(r.boxes) > 0, f"照片里没检到人: {path}"
    xy = r.boxes.xyxy.cpu().numpy()
    k = int(np.argmax(xy[:, 3] - xy[:, 1]))
    x0, y0, x1, y1 = xy[k].astype(int)
    crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    t = TF(Image.fromarray(crop)).unsqueeze(0).cuda()
    with torch.no_grad():
        f = model(t)
    return (f / f.norm(dim=1, keepdim=True))[0].cpu().numpy()


def build_gallery():
    """四段合并档案 -> 扁平列表 [(seg, gid, emb_matrix), ...]"""
    gal = []
    for seg in SEGMENTS:
        z = np.load(os.path.join(IDX_ROOT, seg, "index_merged.npz"))
        for key in z.files:
            if key.startswith("emb_"):
                gal.append((seg, int(key.split("_")[1]), z[key]))
    return gal


def search(qfeats, gal, fusion):
    """返回按相似度降序的 [(seg, gid, sim)]。fusion: max 逐查询取最大 / mean 先平均。"""
    res = []
    if fusion == "mean":
        q = qfeats.mean(axis=0)
        q /= np.linalg.norm(q) + 1e-8
        for seg, gid, emb in gal:
            res.append((seg, gid, float((emb @ q).max())))
    else:
        for seg, gid, emb in gal:
            res.append((seg, gid, float((emb @ qfeats.T).max())))
    return sorted(res, key=lambda x: -x[2])


def rank_of(ranked, seg, gid):
    for i, (s, g, _) in enumerate(ranked):
        if s == seg and g == gid:
            return i + 1
    return None


def main():
    model, yolo = load_reid(), YOLO(YOLO_W)
    photos = sorted(glob.glob(os.path.join(QUERY_DIR, "*.jpg")))
    print(f"查询照片 {len(photos)} 张")
    qall = np.stack([query_feat(model, yolo, p) for p in photos])
    gal = build_gallery()
    print(f"gallery：{len(gal)} 个组档案（跨 {len(SEGMENTS)} 段视频）\n")

    runs = [("B1 单图max", qall[:1], "max"), ("B2 三图max", qall[:3], "max"),
            ("B3 四图max", qall[:4], "max"), ("B4 四图mean", qall[:4], "mean")]

    rows, print_hdr = [], None
    print(f"{'策略':<12} {'hit@1(严/宽)':<14} {'hit@3(严/宽)':<14} "
          f"{'hit@5(严/宽)':<14} {'目标组名次明细(严格口径)'}")
    for name, qf, fusion in runs:
        ranked = search(qf, gal, fusion)
        r_strict = [rank_of(ranked, s, g) for s, gs in TARGETS_STRICT.items() for g in gs]
        r_len = [rank_of(ranked, s, g) for s, gs in TARGETS_LENIENT.items() for g in gs]
        def hits(rs, k):
            return sum(1 for r in rs if r is not None and r <= k)
        line = (f"{name:<12} {hits(r_strict,1)}/{hits(r_len,1)}/4"
                f"{' ' * 4}{hits(r_strict,3)}/{hits(r_len,3)}/4"
                f"{' ' * 4}{hits(r_strict,5)}/{hits(r_len,5)}/4"
                f"{' ' * 4} " + " ".join(f"{s}:r{r}" for s, r in
                                          zip(list(TARGETS_STRICT), r_strict)))
        print(line)
        top10 = [(s, g, round(v, 4)) for s, g, v in ranked[:10]]
        rows.append((name, line, top10))

    print("\n--- B3 四图max Top-10 明细 ---")
    for i, (s, g, v) in enumerate(rows[2][2]):
        mark = " <= 目标" if g in TARGETS_STRICT[s] else ""
        print(f"  #{i+1:>2} {s}/G{g}  sim={v:.4f}{mark}")

    with open(CSV_OUT, "w", encoding="utf-8-sig") as f:
        f.write("strategy,hit1_strict,hit1_lenient,hit3_strict,hit3_lenient,"
                "hit5_strict,hit5_lenient,top10\n")
        for name, _, top10 in rows:
            f.write(f"{name}," + ",".join(str(x) for x in
                    [_hit_for(rows, name, runs, k, strict) for k in (1, 3, 5)
                     for strict in (True, False)])
                    + ',"' + str(top10).replace('"', "'") + '"\n')
    print(f"\n=> CSV 已写 {CSV_OUT}")


def _hit_for(rows, name, runs, k, strict):
    """从已打印逻辑重算（保持单一实现源）：按 name 找回其 ranked 再数命中"""
    qf, fusion = [(q, fu) for n, q, fu in runs if n == name][0]
    ranked = search(qf, build_gallery(), fusion)
    tg = TARGETS_STRICT if strict else TARGETS_LENIENT
    rs = [rank_of(ranked, s, g) for s, gs in tg.items() for g in gs]
    return sum(1 for r in rs if r is not None and r <= k)


if __name__ == "__main__":
    main()
