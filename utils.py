import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_samples(generator, fixed_z, epoch, save_dir, device):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_z).cpu()
    generator.train()

    fake = fake * 0.5 + 0.5
    fake = fake.clamp(0, 1)

    n = fixed_z.size(0)
    cols = int(np.sqrt(n))

    fig, axes = plt.subplots(cols, cols, figsize=(cols, cols))
    for i in range(cols):
        for j in range(cols):
            idx = i * cols + j
            img = fake[idx].permute(1, 2, 0).numpy()
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
    plt.tight_layout()
    path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_losses(g_losses, d_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss", alpha=0.8)
    plt.plot(d_losses, label="Discriminator Loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(path, dpi=100)
    plt.close()
