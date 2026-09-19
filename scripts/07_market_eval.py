# -*- coding: utf-8 -*-
"""
07 · Market 小批量评估：提特征 -> 余弦比对 -> 排序 -> 判分
骨架已搭好，只留 6 个 TODO 空格，逐个填完后 python scripts/07_market_eval.py 能跑通即验收。
每填完一个 TODO 就单测那一个函数（文件末尾有各自的测试入口），别攒到最后一起调。
"""
import glob
import random

import torch
import torchreid
from PIL import Image
from torchvision import transforms

# ---------- 常数（已定死，不用改） ----------
DATA_ROOT = r"E:\QianwenApp\workplaces\视频目标人物检索项目\data\market1501\Market-1501-v15.09.15"
QUERY_DIR = DATA_ROOT + r"\query"
GALLERY_DIR = DATA_ROOT + r"\bounding_box_test"
N_QUERY = 50  # 只取前 50 张 query
N_GALLERY = 2000  # gallery 随机抽 2000 张
SEED = 42  # 固定随机种子，保证复跑数字一致
TOPK = 5  # 每条 query 看前 5 名

NORM_MEAN = [0.485, 0.456, 0.406]  # OSNet 官方训练配方，别换
NORM_STD = [0.229, 0.224, 0.225]

IMG_H, IMG_W = 256, 128  # torchreid 标准输入尺寸(高x宽)


# ---------- TODO-1: 解析文件名 ----------
def parse_name(path):

    fname = path.split("\\")[-1]
    a = fname.split("_")
    pid = int(a[0])
    camid = int(a[1][1])

    return pid, camid  # 返回 (pid, camid)


# ---------- TODO-2: 收集样本列表 ----------
def collect_samples():
    """


    """
    qpaths = sorted(glob.glob(QUERY_DIR + r"\*.jpg"))
    queries = []
    for p in qpaths[:N_QUERY]:
        pid, camid = parse_name(p)
        if pid == -1:            # junk 安检：query 理论无 -1，防御性过滤
            continue
        queries.append((p, pid, camid))

    gpaths = glob.glob(GALLERY_DIR + r"\*.jpg")
    random.seed(SEED)
    gpicks = random.sample(gpaths, N_GALLERY)
    galleries = []
    for p in gpicks:
        pid, camid = parse_name(p)
        if pid == -1:            # 官方协议：junk(-1) 不参与测试，剔除
            continue
        galleries.append((p, pid, camid))

    return queries, galleries


# ---------- TODO-3: 搭模型 ----------
def build_reid_model():
    """返回.eval()状态的 osnet_x1_0 + 官方预训练权重。

    提示：torchreid.models.build_model(name='osnet_x1_0', num_classes=751, pretrained=True)
    第一次运行会自动联网下载权重 —— 卡住/报错就原样截图发指导者（那是 1.13 坑）。
    想清楚为什么这里要 .eval()（提示：dropout/BN 在训练与推理时的行为差异）。
    """
    model = torchreid.models.build_model(name='osnet_x1_0', num_classes=751, pretrained=False)
    # torchreid 内置下载只有 ImageNet 版权重（对 ReID 无判别力，Rank-1 只有 6%）
    # 这里手动加载作者团队训练版 osnet_x1_0_market_250x125x3 权重
    sd = torch.load(r"C:\Users\ssybdx\.cache\torch\checkpoints\osnet_x1_0_market1501.pth", map_location="cpu")
    ret = model.load_state_dict(sd, strict=False)  # 返回未匹配清单，下面核验
    print("=> 未加载的模型层:", list(ret.missing_keys))   # 只允许出现 classifier.*
    print("=> 多余的权重键:", list(ret.unexpected_keys))
    return model.eval()


# ---------- TODO-4: 单图预处理 ----------
PREPROCESS = transforms.Compose([  # 流水线定义一次，反复使用
    transforms.Resize((IMG_H, IMG_W)),  # 道一：统一裁成 256x128
    transforms.ToTensor(),  # 道二：图片 → tensor，像素从0~255缩到0~1
    transforms.Normalize(NORM_MEAN, NORM_STD),  # 道三：按官方配方标准化
])

def load_image(path):
    """读一张 jpg，按官方配方变成 (1,3,256,128) 的 float tensor。"""
    img = Image.open(path).convert('RGB')  # 开图；强制转3通道
    t = PREPROCESS(img)  # 过流水线 → (3,256,128)
    return t.unsqueeze(0)  # 上托盘 → (1,3,256,128)


# ---------- TODO-5: 批量提特征 ----------
@torch.no_grad()
def extract_feats(model, samples, batch_size=64):
    """对 samples 逐批前向，返回 (len(samples), 512) 且【每行 L2 归一化】的 tensor。

    提示：切片取批 -> torch.stack([load_image(p) for p,_,_ in 批]) 成 (B,3,H,W)
    -> model(批) 得 (B,512) -> 存起来；全部做完 torch.cat。
    归一化：torch.nn.functional.normalize(特征, dim=1)  —— 这一行就是"余弦化"的开关。
    """
    feats = []
    for i in range(0, len(samples), batch_size):        # 步长=batch_size 地往前跳
        batch = samples[i:i + batch_size]                # 切出一批（最后可能不足64）
        imgs = torch.stack([load_image(p).squeeze(0)     # 每张 (1,3,H,W) 先捏回 (3,H,W)
                            for p, _, _ in batch])       # stack 成 (B,3,H,W)
        out = model(imgs)                                # 前向 → (B,512)
        feats.append(out)                                # 存进袋子
    all_feats = torch.cat(feats, dim=0)                  # 全部批次拼成 (N,512)
    return torch.nn.functional.normalize(all_feats, dim=1)  # 每行 L2 归一化，"余弦化"开关


# ---------- 主流程（已写好，读懂再跑） ----------
def main():
    queries, galleries = collect_samples()
    model = build_reid_model()
    qf = extract_feats(model, queries)  # (50,512)
    gf = extract_feats(model, galleries)  # (2000,512)

    sim = qf @ gf.T  # 归一化后点积 = 余弦相似度 (50,2000)
    topv, topi = sim.topk(TOPK, dim=1)  # 每条query前5名

    rank1_all, rank1_xcam = 0, 0
    for qi, (qpath, qpid, qcam) in enumerate(queries):
        hits = [(galleries[topi[qi, k]][1], galleries[topi[qi, k]][2], topv[qi, k])
                for k in range(TOPK)]
        # 全量判卷：第一名 pid 相同即命中
        if hits[0][0] == qpid:
            rank1_all += 1
        # 剔除同相机后的判卷：跳过与前5名里 camid==qcam 的，取剩下第一个
        for pid, cam, _ in hits:
            if cam == qcam:
                continue
            if pid == qpid:
                rank1_xcam += 1
            break
        if qi < 2:  # 前两张打印 top-3 明细给你肉眼核对
            print(f"\nquery: {qpath}  (pid={qpid}, cam={qcam})")
            for k in range(3):
                print(f"  top-{k + 1}: pid={hits[k][0]} cam={hits[k][1]} sim={hits[k][2]:.4f}")

    print(f"\nRank-1(含同相机): {rank1_all}/{N_QUERY} = {rank1_all / N_QUERY:.2%}")
    print(f"Rank-1(剔除同相机): {rank1_xcam}/{N_QUERY} = {rank1_xcam / N_QUERY:.2%}")


# ---------- 单函数冒烟入口：python scripts/07_market_eval.py 1 就只测 TODO-1 ----------
if __name__ == "__main__":
    import sys

    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "1":
        print(parse_name(r"x\y\0001_c1s1_001051_03.jpg"))  # 期望: (1, 1)
    elif step == "2":
        q, g = collect_samples()
        print(len(q), len(g))
        print(q[0])
    elif step == "3":
        m = build_reid_model()
        print(type(m).__name__, next(m.parameters()).shape)
    elif step == "4":
        t = load_image(sorted(glob.glob(QUERY_DIR + r"\*.jpg"))[0])
        print(t.shape)  # 期望 torch.Size([1,3,256,128])
    elif step == "5":
        m = build_reid_model()
        one = [(sorted(glob.glob(QUERY_DIR + r"\*.jpg"))[0], 1, 1)]
        print(extract_feats(m, one).shape)  # 期望 torch.Size([1, 512])
    else:
        main()
