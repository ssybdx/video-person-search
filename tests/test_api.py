# -*- coding: utf-8 -*-
"""14 · 接口冒烟测试（DB_MODE=sqlite 走本地兜底，不依赖 MySQL 服务）

运行: 在「项目根」下 python -m pytest tests/test_api.py -q
"""
import os
import sys

os.environ["DB_MODE"] = "sqlite"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pytest
from fastapi.testclient import TestClient

import db
db.init_db()
import api
m10 = api.m10
m10.FRAME_STEP = 25          # 测试提速：只跟踪10帧（真实值2→125帧要数分钟）

VIDEO = os.path.join(ROOT, "data", "videos", "people.mp4")

client = TestClient(api.app)


@pytest.fixture(scope="module")
def video_id():
    with open(VIDEO, "rb") as f:
        r = client.post("/videos", files={"video": ("people_test.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["num_persons"] >= 1
    return body["video_id"]


def test_upload_video(video_id):
    assert video_id >= 1


def test_list_videos(video_id):
    r = client.get("/videos")
    assert r.status_code == 200
    assert any(v["id"] == video_id for v in r.json())


def test_search(video_id):
    # 查询图用档案库里某人crop（自查询，应Top1命中该ID）
    lib = api._STATE["db_by_video"][video_id]
    tid_true, item = next(iter(lib.items()))
    files = {"imgs": (item["paths"][0], open(item["paths"][0], "rb"), "image/jpeg")}
    r = client.post("/search",
                    data={"video_id": video_id, "fusion": "max", "topk": "5"},
                    files=files)
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert res[0]["track_id"] == tid_true
    assert res[0]["score"] > 0.9


def test_search_rejects_bad_ext(video_id):
    r = client.post("/search", json={"video_id": video_id}, files=[])
    assert r.status_code in (400, 422)
