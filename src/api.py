# -*- coding: utf-8 -*-
"""14 · FastAPI 服务：POST /videos 建索引；POST /search 照片查询；GET /videos 列表

运行: python -m uvicorn src.api:app --port 8000  （项目根目录下）
"""
import os
import sys
import time
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

import db
m07 = __import__("07_market_eval")
m10 = __import__("10_index_pipeline")
m11 = __import__("11_query_search")

app = FastAPI(title="Video Person Retrieval API", version="0.1")

_STATE = {"reid": None, "db_by_video": {}}


def _reid():
    if _STATE["reid"] is None:
        _STATE["reid"] = m07.build_reid_model()
    return _STATE["reid"]


def _index_npz_for(video_id):
    row = db.get_video(video_id)
    if not row or not row["index_npz"]:
        raise HTTPException(404, f"video {video_id} 不存在或尚未建索引")
    return row


class SearchBody(BaseModel):
    video_id: int
    fusion: str = "max"
    topk: int = 5


@app.post("/videos")
async def upload_video(video: UploadFile = File(...)):
    t0 = time.time()
    if not video.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(400, "仅支持 mp4/avi/mov")
    tmp = os.path.join(ROOT, "data", "videos", video.filename)
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(await video.read())

    frames = m10.read_frames(tmp)
    tracks = m10.track_video(m10.YOLO(m10.WEIGHTS), frames)
    stem = os.path.splitext(video.filename)[0]
    vid_dir = os.path.join(ROOT, "data", "index", f"video_{stem}")
    npz = os.path.join(vid_dir, "index.npz")
    out_dir_backup = m10.OUT_DIR
    m10.OUT_DIR = vid_dir                  # 该视频的档案库与crops全部隔离在子目录内
    try:
        m10.build_index(frames, tracks, _reid())
    finally:
        m10.OUT_DIR = out_dir_backup

    lib = m11.load_index(npz)
    vid = db.insert_video(video.filename, tmp, duration_s=len(frames) * 2 / 25.0,
                          num_persons=len(lib), index_npz=npz)
    for tid, item in lib.items():
        db.insert_person(vid, tid, int(item["frames"].min()), int(item["frames"].max()),
                         item["frames"].min() * 2 / 25.0, item["frames"].max() * 2 / 25.0,
                         len(item["paths"]), f"emb_{tid}", "|".join(item["paths"]))
    _STATE["db_by_video"][vid] = m11.load_index(npz)
    return {"video_id": vid, "num_persons": len(lib), "elapsed_s": round(time.time() - t0, 1)}


@app.post("/search")
async def search(video_id: int = Form(...), fusion: str = Form("max"), topk: int = Form(5),
                 imgs: list[UploadFile] = File(...)):
    if not imgs:
        raise HTTPException(400, "至少1张查询图")
    row = _index_npz_for(video_id)
    lib = _STATE["db_by_video"].get(video_id) or m11.load_index(row["index_npz"])
    tmp_paths = []
    for im in imgs:
        fd, p = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        with open(p, "wb") as f:
            f.write(await im.read())
        tmp_paths.append(p)
    t0 = time.time()
    try:
        embs = m11.embed_query_images(_reid(), tmp_paths)
        res = m11.search_core(embs, lib, fusion=fusion, k=topk)
    finally:
        for p in tmp_paths:
            os.remove(p)
    ms = int((time.time() - t0) * 1000)
    top = res[0] if res else None
    db.insert_search_log("|".join(i.filename for i in imgs), fusion,
                         top[0] if top else None, top[1] if top else None,
                         hit=False, elapsed_ms=ms)   # hit 由前端人工判卷后不回写（demo口径）
    return {"results": [dict(track_id=t, score=s, frames=f, crop=cp)
                        for t, s, f, cp, _ in res]}


@app.get("/videos")
async def videos():
    return db.list_videos()
