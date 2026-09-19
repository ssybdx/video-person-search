# -*- coding: utf-8 -*-
"""15 · Streamlit 前端：上传视频建索引 / 上传照片查询 / 结果时间轴+代表截图

启动(项目根，两个终端):
  python -m uvicorn src.api:app --port 8000
  python -m streamlit run web/app.py
"""
import os
import tempfile
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="视频目标人物检索", layout="wide")
st.title("🎥 视频目标人物检索系统")

tab_idx, tab_q = st.tabs(["① 视频入库", "② 照片查询"])

with tab_idx:
    st.subheader("上传监控/街景视频，自动构建人员档案库")
    vf = st.file_uploader("选择视频 (mp4/avi/mov)", type=["mp4", "avi", "mov"], key="vid")
    if st.button("建索引", disabled=vf is None):
        st.info("CPU 建索引约需几分钟，请耐心等待…")
        r = requests.post(f"{API}/videos", files={"video": (vf.name, vf.getvalue())}, timeout=3600)
        (st.success if r.ok else st.error)(r.text)
    try:
        vids = requests.get(f"{API}/videos", timeout=10).json()
        if vids:
            st.dataframe(vids)
    except Exception:
        st.warning("后端未启动？先运行 uvicorn src.api:app --port 8000")

with tab_q:
    st.subheader("上传目标人物 1~N 张全身照片，检索是谁、何时出现")
    try:
        vids = requests.get(f"{API}/videos", timeout=10).json()
    except Exception:
        vids = []
    if not vids:
        st.info("还没有已入库视频，请先在『视频入库』上传。")
    else:
        sel = st.selectbox("选择视频", vids, format_func=lambda v: f"#{v['id']} {v['filename']}（{v['num_persons']}人）")
        imgs = st.file_uploader("查询照片（可多选，建议同一人不同角度全身照）",
                                type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        fusion = st.radio("融合策略", ["max", "mean"], horizontal=True)
        if st.button("开始检索", disabled=not imgs):
            files = {"imgs": [(i.name, i.getvalue()) for i in imgs]}
            r = requests.post(f"{API}/search",
                              data={"video_id": sel["id"], "fusion": fusion, "topk": "5"},
                              files=files, timeout=600)
            if not r.ok:
                st.error(r.text)
            else:
                for res in r.json()["results"]:
                    tid, score = res["track_id"], res["score"]
                    st.markdown(f"### Top: ID {tid} — 相似度 {score:.3f}")
                    fr = res["frames"]
                    st.caption(f"出现帧号: {fr[0]} ~ {fr[-1]}（25fps 约 {fr[0]/25:.1f}s ~ {fr[-1]/25:.1f}s）")
                    with st.expander("代表截图", expanded=True):
                        cp = res["crop"]
                        if os.path.exists(cp):
                            st.image(cp, caption=os.path.basename(cp), width=160)
                        else:
                            st.text(cp)
