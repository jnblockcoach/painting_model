# CIFAR-10 DCGAN

Image generation model (~1M params) trained on CIFAR-10 with PyTorch.

## 项目结构

```
├── config.py          # 超参数配置
├── model.py           # Generator (378K) + Discriminator (691K)
├── dataset.py         # CIFAR-10 DataLoader
├── train.py           # 训练主循环
├── test.py            # 测试生成图片
├── utils.py           # 保存样本 + loss 曲线
├── data/
│   └── cifar-10-batches-py/
├── samples/           # 训练中每 epoch 生成的样本
├── test_output/       # test.py 生成的结果
├── checkpoints/       # 模型权重
└── logs/              # loss 记录
```

## 训练

```powershell
py -3.11 train.py
```

每 epoch 打印进度条（G loss / D loss），每 10 epoch 保存 checkpoint，G loss 创新低时自动保存 `G_best.pth`。

## 测试

```powershell
# 8x8 网格（默认 G_best.pth → G_final.pth → 最新 epoch）
py -3.11 test.py

# 指定 checkpoint
py -3.11 test.py --ckpt checkpoints/G_epoch_050.pth

# 单张图
py -3.11 test.py --mode single

# 指定输出路径
py -3.11 test.py --output my_test.png
```

结果保存在 `test_output/` 目录。

## 配置

修改 `config.py` 可调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `G_FEATURES` | 128 | Generator 通道基数 |
| `D_FEATURES` | 32 | Discriminator 通道基数 |
| `Z_DIM` | 100 | 噪声向量维度 |
| `BATCH_SIZE` | 128 | 批大小 |
| `EPOCHS` | 100 | 训练轮数 |
| `G_LR` / `D_LR` | 2e-4 | 学习率 |

## 参数量

| 模型 | 参数量 |
|------|--------|
| Generator | 377,744 |
| Discriminator | 690,816 |
| 合计 | 1,068,560 |
