# 基于多图查询的视频目标人物检索系统

> 输入目标人物 1~N 张全身照片 → 分析监控/街景视频 → 返回"是谁（跟踪ID）、几点到几点出现过、走过哪些位置"。
> 技术链：YOLOv8 检测 → ByteTrack 跟踪 → 自训练 OSNet 特征 → 多图 max 融合检索。
> ⚠️ 演示视频与档案数据因隐私/体积不入库；指标数字为占位（XX），以阶段 2/3 实测回填。

## 架构

```
[索引端]  视频 ─抽帧─ YOLOv8(行人检测) ─ ByteTrack(跨帧ID) ─ 每ID挑清晰帧裁剪
          ─ OSNet 提特征(每人5向量组) ─ index.npz 档案库 + MySQL(videos/persons)

[查询端]  照片×N ─ OSNet 提特征 ─ 与档案库全向量比对(余弦)
          ─ max/mean 融合 ─ Top-5 排序 ─ 命中ID + 时段 + 代表截图
```

（架构图待补：docs/architecture.png）

## 快速开始

```bash
# 1. 环境（Python 3.10, CPU/CUDA 自适应）
conda create -n vperson python=3.10 && conda activate vperson
pip install torch torchvision ultralytics          # CUDA 版见官方站点
pip install gdown yacs future imagechardet tensorboard

# 2. torchreid（源码安装，两处适配见 docs/踩坑记录）
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid   # 需删 setup.py ext_modules 行（无编译器场景）
pip install . --no-build-isolation --no-deps

# 3. 启动
python -m uvicorn src.api:app --port 8000          # 终端1
python -m streamlit run web/app.py                 # 终端2
```

## 使用

- 打开 Streamlit → 「视频入库」上传 mp4 → 等待建索引
- 「照片查询」选视频 + 传 1~N 张全身照 → 返回 Top-5 命中（ID/相似度/出现时段/代表截图）
- API：`POST /videos`（multipart 上传视频建索引）、`POST /search`（Form: video_id/fusion/topk + 图片）、`GET /videos`

## 性能指标

| 项 | 数值 |
|---|---|
| Market-1501 预训练 OSNet Rank-1 / mAP | XX% / XX%（小批量管线验证 48%，全量待 GPU） |
| 自训练 OSNet Rank-1 / mAP | XX% / XX%（阶段 2） |
| 测试视频 Top-1 命中率（单图→5图 max 融合） | XX% → XX%（+XX pp） |
| max vs mean 融合 | XX% vs XX% |

（GPU 到位后按 docs/training_plan.md 消融矩阵 A0~A3、B1~B5 回填）

## 代码结构

```
scripts/07_market_eval.py     Market 小批量检索评估管线（检索端雏形）
scripts/10_index_pipeline.py  索引端：抽帧→检测→跟踪→留底→建库
scripts/11_query_search.py    查询端：照片→特征→融合比对→Top-k
scripts/12_mot16_to_video.py  MOT16 帧序列转视频工具
scripts/08_train_smoke.py     训练链 CPU 冒烟（数据→前向→联合损失→反向）
src/fusion.py                 max/mean 融合策略（含单元测试，毕设创新点）
src/db.py                     数据访问层（MySQL/SQLite 双模式）
src/api.py                    FastAPI 服务
web/app.py                    Streamlit 前端
tests/                        pytest：fusion 7项 + api 4项全绿
docs/                         训练与消融计划 / 过程记录
data/                         不入库（数据集、视频、档案库）
```

## 已知边界（诚实声明）

- 认衣着体态、**不认脸**：查询照片需人体（最好全身），大头照无效
- **换装会失效**：cloth-changing 是领域开放难题
- 日常照片查监控存在 domain gap，实验室指标会回落

## License

MIT（第三方数据集/模型权重按各自协议：Market-1501 CC BY-NC-SA 4.0，ultralytics GPL-3.0）
