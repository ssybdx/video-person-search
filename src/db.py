# -*- coding: utf-8 -*-
"""数据库适配层：优先 MySQL(本机开发库)，连接失败自动回退 SQLite。

上层只认 5 个函数，不关心底层是什么。.env 里的 DB_* 变量可覆盖默认值。
"""
import os
import sqlite3

DB_MODE = os.environ.get("DB_MODE", "auto")      # auto | mysql | sqlite
SQLITE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vperson.db")
MYSQL_CONF = dict(host=os.environ.get("DB_HOST", "127.0.0.1"), port=int(os.environ.get("DB_PORT", 3306)),
                  user=os.environ.get("DB_USER", "vperson"),
                  password=os.environ.get("DB_PASS", ""),  # 密码走环境变量；留空时 MySQL 连不上会自动回退 SQLite
                  database=os.environ.get("DB_NAME", "vperson"),
                  charset="utf8mb4")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY {ai},
    filename TEXT NOT NULL, abs_path TEXT NOT NULL,
    duration_s REAL, num_persons INTEGER DEFAULT 0,
    index_npz TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY {ai},
    video_id INTEGER NOT NULL, track_id INTEGER NOT NULL,
    start_frame INTEGER, end_frame INTEGER,
    start_time_s REAL, end_time_s REAL,
    num_crops INTEGER, emb_key TEXT, crop_paths TEXT);
CREATE TABLE IF NOT EXISTS search_logs (
    id INTEGER PRIMARY KEY {ai},
    query_paths TEXT NOT NULL, fusion TEXT DEFAULT 'max',
    top_track_id INTEGER, top_score REAL, hit INTEGER,
    elapsed_ms INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
"""


def _connect():
    if DB_MODE in ("auto", "mysql"):
        try:
            import pymysql
            return pymysql.connect(**MYSQL_CONF), "mysql"
        except Exception:
            if DB_MODE == "mysql":
                raise
    return sqlite3.connect(SQLITE_PATH), "sqlite"


def get_conn():
    return _connect()


def init_db():
    conn, mode = get_conn()
    cur = conn.cursor()
    ai = "AUTO_INCREMENT" if mode == "mysql" else "AUTOINCREMENT"
    for stmt in [s for s in _SCHEMA.format(ai=ai).split(";") if s.strip()]:
        cur.execute(stmt)
    conn.commit(); conn.close()
    return mode


def insert_video(filename, abs_path, duration_s, num_persons, index_npz):
    conn, mode = get_conn(); cur = conn.cursor()
    ph = "%s" if mode == "mysql" else "?"
    cur.execute(f"INSERT INTO videos (filename,abs_path,duration_s,num_persons,index_npz) "
                f"VALUES ({ph},{ph},{ph},{ph},{ph})",
                (filename, abs_path, duration_s, num_persons, index_npz))
    conn.commit(); vid = cur.lastrowid; conn.close()
    return vid


def insert_person(video_id, track_id, start_frame, end_frame, start_time_s, end_time_s,
                  num_crops, emb_key, crop_paths):
    conn, mode = get_conn(); cur = conn.cursor()
    ph = "%s" if mode == "mysql" else "?"
    cur.execute(f"INSERT INTO persons (video_id,track_id,start_frame,end_frame,start_time_s,"
                f"end_time_s,num_crops,emb_key,crop_paths) VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
                (video_id, track_id, start_frame, end_frame, start_time_s, end_time_s,
                 num_crops, emb_key, crop_paths))
    conn.commit(); cur_pid = cur.lastrowid; conn.close()
    return cur_pid


def insert_search_log(query_paths, fusion, top_track_id, top_score, hit, elapsed_ms):
    conn, mode = get_conn(); cur = conn.cursor()
    ph = "%s" if mode == "mysql" else "?"
    cur.execute(f"INSERT INTO search_logs (query_paths,fusion,top_track_id,top_score,hit,elapsed_ms) "
                f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph})",
                (query_paths, fusion, top_track_id, top_score, int(hit), elapsed_ms))
    conn.commit(); cur.close(); conn.close()


def list_videos():
    conn, mode = get_conn(); cur = conn.cursor()
    cur.execute("SELECT id, filename, num_persons, created_at FROM videos ORDER BY id DESC")
    rows = cur.fetchall(); conn.close()
    return [dict(id=r[0], filename=r[1], num_persons=r[2], created_at=str(r[3])) for r in rows]


def get_video(video_id):
    conn, mode = get_conn(); cur = conn.cursor()
    ph = "%s" if mode == "mysql" else "?"
    cur.execute(f"SELECT id, filename, abs_path, index_npz, num_persons FROM videos WHERE id={ph}",
                (video_id,))
    r = cur.fetchone(); conn.close()
    if not r:
        return None
    return dict(id=r[0], filename=r[1], abs_path=r[2], index_npz=r[3], num_persons=r[4])
