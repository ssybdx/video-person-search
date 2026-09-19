# -*- coding: utf-8 -*-
"""08 · 训练脚本冒烟：CPU 上完整走 1 个 step（前向+联合损失+反向+更新），验证训练链无报错。

这不是正式训练（正式训练等 3060 Ti，命令见 docs/training_plan.md）。
冒烟只证明：数据管线、模型、loss、优化器四件套在本环境版本组合下能跑。
运行: python scripts/08_train_smoke.py
"""
import os
import sys
import glob
import random

import torch
import torchreid
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TinyMarket(Dataset):
    """从 bounding_box_train 抽 N 张图，pid 取自文件名——最小可跑训练集。"""
    def __init__(self, root, num_pid=8, per_pid=4):
        paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        groups = {}
        for p in paths:
            pid = int(os.path.basename(p).split("_")[0])
            if pid > 0:
                groups.setdefault(pid, []).append(p)
        pids = sorted(groups)
        random.seed(42)
        chosen = random.sample(pids, num_pid)
        self.num_pids = len(pids)
        self.items = []
        for label, pid in enumerate(chosen):          # 重编号 0..num-1 供 CE 用
            for p in groups[pid][:per_pid]:
                self.items.append((p, label))
        self.tf = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, pid = self.items[i]
        from PIL import Image
        return self.tf(Image.open(p).convert("RGB")), pid


def main():
    data_root = r"E:\QianwenApp\workplaces\视频目标人物检索项目\data\market1501\Market-1501-v15.09.15"
    ds = TinyMarket(os.path.join(data_root, "bounding_box_train"))   # 8pid×4张=32
    num_pids = ds.num_pids

    model = torchreid.models.build_model(name="osnet_x1_0", num_classes=num_pids,
                                         loss="triplet", pretrained=True)   # 见 osnet.py:435-436: 返回(logits, feat)
    opt = torchreid.optim.build_optimizer(model, lr=0.01)
    triplet = torchreid.losses.TripletLoss(margin=0.3)   # 内置 hard mining，CPU 可用
    # ⚠ torchreid 的 CrossEntropyLoss 硬编码 targets.cuda()（见其源码，CPU 冒烟必炸），
    #   冒烟用等价 torch.nn 实现；GPU 正式训练走 torchreid engine 不受此影响
    ce_fn = torch.nn.CrossEntropyLoss()

    dl = DataLoader(ds, batch_size=len(ds), shuffle=False)   # 一个 batch 全覆盖
    x, pid = next(iter(dl))
    model.train()
    logits, feat = model(x)
    loss_t = triplet(feat, pid)          # 特征上挖三元组
    loss_c = ce_fn(logits, pid)          # 分类头交叉熵
    loss = loss_t + loss_c
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"冒烟通过: batch={tuple(x.shape)} logits={tuple(logits.shape)} feat={tuple(feat.shape)} "
          f"loss(triplet+ce)={loss_t.item():.3f}+{loss_c.item():.3f}")
    print("四件套 OK：数据->前向->联合损失->反向更新")


if __name__ == "__main__":
    main()
