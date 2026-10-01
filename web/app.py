# -*- coding: utf-8 -*-
"""19 · Streamlit 前端（真实链路版）：照片查询 -> 跨段档案检索 -> 时段 + 证据截图

与 9 月 mock 版的区别：不再走 uvicorn+src/api 的旧索引，直接复用
scripts/16_b_group_ablation.py 的已验证实现（query_feat/build_gallery/search），
对接 data/index_real/*/index_merged.npz（自训 A2 权重 + 轨迹合并后的真实档案库）。

启动（PyCharm Terminal，项目根）:
  conda activate vperson
  python -m streamlit run web/app.py
"""
import glob
import importlib
import json
import os
import sys

import cv2
import numpy as np
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
m16 = importlib.import_module("16_b_group_ablation")   # 复用已验证的查询/检索实现

IDX_ROOT = os.path.join(ROOT, "data", "index_real")
GT_PATH = os.path.join(IDX_ROOT, "ground_truth.json")
DEMO_PHOTOS = sorted(glob.glob(os.path.join(m16.QUERY_DIR, "*.jpg")))


def imread_cn(path):
    if not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


@st.cache_resource(show_spinner="加载自训模型（约10秒）…")
def get_engines():
    return m16.load_reid(), m16.YOLO(m16.YOLO_W)


@st.cache_resource(show_spinner="加载档案库…")
def get_gallery():
    return m16.build_gallery()


st.set_page_config(page_title="视频目标人物检索", layout="wide")
st.title("视频目标人物检索系统")
st.caption("检测(YOLOv8) → 跟踪(ByteTrack) → 特征(自训OSNet-A2) → 多图max融合检索")

tab_q, tab_lib = st.tabs(["① 照片检索", "② 档案库总览"])

with tab_q:
    st.subheader("上传目标人物全身照（可多张），检索是谁、在哪些视频、几点到几点出现")
    col_l, col_r = st.columns([1, 1])
    with col_l:
        use_demo = st.checkbox("使用演示照片（人物A 的 4 张）", value=bool(DEMO_PHOTOS))
        ups = st.file_uploader("或上传照片 (jpg/png，同一人不同角度)", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True)
        fusion = st.radio("融合策略", ["max（推荐）", "mean"], horizontal=True)
        segs = st.multiselect("检索范围", m16.SEGMENTS, default=m16.SEGMENTS)
        go = st.button("开始检索", type="primary")
    with col_r:
        st.markdown("**预览查询图**")
        if use_demo and DEMO_PHOTOS:
            show = [imread_cn(p) for p in DEMO_PHOTOS[:4]]
        elif ups:
            show = [cv2.imdecode(np.frombuffer(u.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    for u in ups[:4]]
        else:
            show = []
        show = [s for s in show if s is not None]
        if show:
            cols = st.columns(min(4, len(show)))
            for c, im in zip(cols, show):
                c.image(im[:, :, ::-1], width=140)

    if go:
        paths = None
        if use_demo and DEMO_PHOTOS and not ups:
            paths = DEMO_PHOTOS
        elif ups:
            # 上传件落盘后走同一裁剪管线（query_feat 需要文件路径）
            tmp_dir = os.path.join(ROOT, "data", "demo", "web_upload")
            os.makedirs(tmp_dir, exist_ok=True)
            paths = []
            for u in ups:
                p = os.path.join(tmp_dir, u.name)
                with open(p, "wb") as f:
                    f.write(u.getbuffer())
                paths.append(p)
        else:
            st.error("请先勾选演示照片或上传照片")
            paths = None
        if paths:
            model, yolo = get_engines()
            with st.spinner("提取照片特征（人体裁剪）…"):
                qfeats = np.stack([m16.query_feat(model, yolo, p) for p in paths])
            gal = get_gallery()
            with st.spinner("检索档案库…"):
                ranked = m16.search(qfeats, gal, "max" if fusion.startswith("max") else "mean")
            for seg in segs:
                mm_path = os.path.join(IDX_ROOT, seg, "merged_meta.json")
                mm = json.load(open(mm_path, encoding="utf-8"))
                dur = mm["duration_sec"]
                z = np.load(os.path.join(IDX_ROOT, seg, "index_merged.npz"))
                top = [(g, v) for s, g, v in ranked if s == seg][:3]
                st.markdown(f"#### {seg}.mp4（档案 {mm['merge']['n_after']} 人 / 全长 {dur:.1f}s）")
                if not top:
                    st.info("该段无档案")
                    continue
                cols = st.columns(3)
                for c, (g, v) in zip(cols, top):
                    tm = mm["ids"][str(g)]
                    f_s, l_s = tm["first_sec"], tm["last_sec"]
                    c.metric(f"ID {g}", f"相似度 {v:.3f}")
                    # 时间轴：彩色条表示出现时段
                    frac0, frac1 = f_s / dur, l_s / dur
                    bar = (f"<div style='background:#26262E;border-radius:4px;height:14px;"
                           f"position:relative'><div style='position:absolute;left:{frac0*100:.1f}%;"
                           f"width:{max(0.5, (frac1-frac0)*100):.1f}%;top:0;bottom:0;"
                           f"background:#F59E0B;border-radius:4px'></div></div>"
                           f"<div style='color:#71717A;font-size:12px'>{f_s:.1f}s ~ {l_s:.1f}s"
                           f"（{tm['n_track_hits']}帧命中）</div>")
                    c.markdown(bar, unsafe_allow_html=True)
                    ps = [str(p) for p in z[f"paths_{g}"]]
                    ims = [imread_cn(p) for p in ps[:4]]
                    ims = [i[:, :, ::-1] for i in ims if i is not None]
                    if ims:
                        c.image(ims, width="stretch")
                st.divider()

with tab_lib:
    st.subheader("离线建库产物总览（scripts/13_real_index.py + 14_merge_tracks.py）")
    gt = json.load(open(GT_PATH, encoding="utf-8")) if os.path.exists(GT_PATH) else {}
    rows = []
    for seg in m16.SEGMENTS:
        p = os.path.join(IDX_ROOT, seg, "merged_meta.json")
        if not os.path.exists(p):
            continue
        mm = json.load(open(p, encoding="utf-8"))
        t = gt.get(seg, {})
        rows.append({"视频": seg, "时长s": mm["duration_sec"], "档案人数": mm["merge"]["n_after"],
                     "权重": mm["reid_weight"], "演示目标组": t.get("target_group", "—"),
                     "目标时段s": str(t.get("time_sec", "—"))})
    st.dataframe(rows, hide_index=True)
    st.caption("入库新视频：把 mp4 放进 data/videos，PyCharm Terminal 跑 "
               "`python scripts/13_real_index.py <文件名>` 再 "
               "`python scripts/14_merge_tracks.py <段名>`；模型档案库会自动更新。")
    if st.button("强制重载档案库（新建库后点我）"):
        st.cache_resource.clear()
        st.rerun()
