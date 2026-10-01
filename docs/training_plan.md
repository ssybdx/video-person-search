# 训练与消融实验计划（阶段 2 · 等 GPU 执行清单）

> 本文档是 08/09 的交付物：3060 Ti 到位（或租卡）当天，照此清单顺序执行即可，零临场决策。
> 冒烟已验证（2026-09-19，CPU）：数据→前向→TripletLoss+CE 联合损失→反向更新 四件套无报错。

## 一、正式训练命令（2026-09-28 重写：仓库已重构，旧 examples/train.py 不存在，见坑 1.22）

新入口 = `scripts/main.py` + `configs/*.yaml` 基座 + 点号参数覆盖。GPU 冒烟已实测通过（1 epoch 3.5 分钟含全量评估，Rank-1 62.1% 起步）。

```bash
cd C:\Users\ssybdx\deep-person-reid
# A2 组（联合损失 = 主训练）：softmax+triplet，官方 amsgrad 配方
python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml --root E:\QianwenApp\workplaces\视频目标人物检索项目\data data.save_dir log/train_A2 train.max_epoch 70 train.batch_size 64 test.eval_freq 10 loss.name triplet loss.triplet.weight_x 1.0 loss.triplet.margin 0.3
```

- 基座 YAML 已含官方 zoo 配方：amsgrad lr 0.0015、stepsize 60、fixbase_epoch 10（前 10 轮只训分类头）——**别覆盖这些**，94.2% 就是这么来的
- `loss.name triplet + weight_x 1.0` = triplet+CE 联合（源码 main.py:92-94 与 engine 逻辑）
- batch 64 在 8G 显存安全；OOM 降 32（不动 lr）
- **时长预估：70 轮 ≈ 2~2.5 小时**（冒烟实测推算）
- 每 10 轮打印 Rank-1/mAP → **截图存 docs/**；结束 checkpoint 在 `log/train_A2\model\`，改名归档为 `osnet_market_A2_日期.pth` 放 `models\`

### 其余消融组的命令差异（A2 达标后再跑）

| 组 | 在 A2 命令基础上改 |
|---|---|
| A0 | `loss.name softmax`（删掉 triplet 两参数），save_dir 改 train_A0 |
| A1 | `loss.triplet.weight_x 0`（纯triplet），save_dir 改 train_A1 |
| A3 | 关 hard mining：新版 TripletLoss 无开关，需改 site-packages torchreid 源码——**A2 达标后再讨论，不阻塞主线** |

## 二、验收标准

| 指标 | 合格线 | 论文参考 | 不达标动作 |
|---|---|---|---|
| Rank-1 | ≥ 85% | 89% | 先查 lr(0.01→0.005)、epoch(70→100)；两轮不过降到 82%/60% 需与指导者讨论 |
| mAP | ≥ 65% | 76% | 同上 |

**07B（全量评估）先于训练**：用 Market 版权重在 `query=3368 全量 × gallery=19732 全量` 上跑 07 同款脚本（去掉小批量限制），预期 Rank-1 ≈ 80%+——这一步证明"评估管线在官方协议下正确"，是阶段 1B 验收。

## 三、消融实验矩阵（毕设核心章节 + 简历量化数据）

### 3.1 训练侧（阶段 2 产出，每组训一次、测一次）

| 组 | 配置 | 验证假设 |
|---|---|---|
| A0 baseline | softmax only | 基准 |
| A1 | triplet only | 度量学习单独作用 |
| A2 | softmax+triplet（=上面正式训练） | 联合是否最优 |
| A3 | A2 + hard mining 关闭 | hard mining 贡献值 |
| A4 | osnet_x0_75（更窄） | 精度-体积权衡（可选，时间富余才做） |

记录格式（每组一行，填进 docs/ablation_results.csv）：
`组名, 日期, Rank-1, Rank-5, Rank-10, mAP, 训练时长, checkpoint文件名`

### 3.2 检索侧（阶段 3B 产出，用训好的最优权重跑）

| 组 | 查询图数 | 融合策略 | 关注指标 |
|---|---|---|---|
| B1 | 1（单图） | max | Top-1 命中率（基准线） |
| B2 | 3 | max | 多图提升幅度 |
| B3 | 5 | max | 边际收益 |
| B4 | 5 | mean | 两策略对比（糊图敏感性） |
| B5 | 1 | 每人1向量（建库时平均） | "多向量留底"本身的贡献 |

**B5 需要建库时多存一份"平均向量"变体**（10 的 build_index 里加一个开关，现在不动，等 3B 时实现——已记入待办）。
数据源：PRW（自带标注，首选）→ 实拍素材到位后换入验证。

### 3.3 预期结论方向（写论文假设，不预填数字）

- 多图 max 显著优于单图（每加一张约 +3~8pp，实测为准）
- max 对查询图质量不均更鲁棒；mean 受糊图拖累（12 的单元测试已复现该机制）
- 多向量档案（B1 vs B5）即使单查询图也占优

## 四、GPU 到位日执行顺序

1. 装 CUDA 环境（踩坑库 2.4）→ 跑 `torch.cuda.is_available()` 为 True
2. `07_market_eval.py` 全量版（07B）→ 打印 Rank-1 对表官方 ≈80% → **阶段 1B 验收**
3. 训练 A0~A3 四组 → `docs/ablation_results.csv` 填表 → **阶段 2 验收**
4. 最佳权重换进 11/12（`build_reid_model` 一行切换 checkpoint）→ 跑 3B 消融 → **阶段 3B 验收**
5. 之后才轮到演示 GIF / README 终稿数字

## 五、冒烟遗留说明

08 冒烟脚本用 `loss='triplet'` 让 OSNet 返回 `(logits, feat)` 双路（源码依据 `osnet.py:435-436`）；CE 因库内硬编码 cuda（坑 1.18）暂以等价 `torch.nn.CrossEntropyLoss` 验证通路——正式训练走 `examples/train.py` 的 engine，在 GPU 上不受此影响。
