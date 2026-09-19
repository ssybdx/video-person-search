# 训练与消融实验计划（阶段 2 · 等 GPU 执行清单）

> 本文档是 08/09 的交付物：3060 Ti 到位（或租卡）当天，照此清单顺序执行即可，零临场决策。
> 冒烟已验证（2026-09-19，CPU）：数据→前向→TripletLoss+CE 联合损失→反向更新 四件套无报错。

## 一、正式训练命令（换卡日直接用）

```bash
# 前提：CUDA 版 torch（装法见踩坑库 2.4；本机 torchreid 的 CE 硬编码 cuda，
# 交叉验证过必须 GPU——见坑 1.18）
cd <repo>/deep-person-reid
python examples/train.py \
  --root E:\QianwenApp\workplaces\视频目标人物检索项目\data \
  --datasets market1501 \
  --arch osnet_x1_0 \
  -s autodetect \
  --loss softmax triplet \
  --triplet-margin 0.3 \
  --max-epoch 70 \
  --train-batch-size 64 --test-batch-size 64 \
  --height 256 --width 128 \
  --gpu 0 \
  --eval-freq 10 \
  --log-dir logs/train_$(date +%Y%m%d)
```

- 3060 Ti 8G 上 batch 64 + 256×128 是安全配置；OOM 就降 32 并开 `--fp16`（torchreid 支持需确认版本，不支持就降分辨率 128×64——会牺牲精度，记录在案）
- **eval-freq 10**：第 10/20/.../70 epoch 输出 Rank-1/mAP，曲线素材截图保存
- **checkpoint**：torchreid 自动存 `logs/train_*/`，每天训练完立即按 `osnet_market_YYYYMMDD_ep70.pth` 改名归档（毕设要用，交接文档纪律）

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
