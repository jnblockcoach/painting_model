import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DEVICE, Z_DIM, G_FEATURES, CHANNELS, CKPT_DIR, TEST_DIR
from model import Generator


def load_generator(checkpoint_path):
    netG = Generator(Z_DIM, G_FEATURES, CHANNELS).to(DEVICE)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    netG.eval()
    return netG


def generate_grid(netG, n=64, save_path=None):
    cols = int(np.sqrt(n))
    noise = torch.randn(n, Z_DIM, device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        fake = netG(noise).cpu()

    fake = fake * 0.5 + 0.5
    fake = fake.clamp(0, 1)

    fig, axes = plt.subplots(cols, cols, figsize=(cols, cols))
    for i in range(cols):
        for j in range(cols):
            idx = i * cols + j
            img = fake[idx].permute(1, 2, 0).numpy()
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def generate_single(netG, save_path=None):
    noise = torch.randn(1, Z_DIM, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        img = netG(noise).cpu()[0]
    img = img * 0.5 + 0.5
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test trained generator")
    parser.add_argument("--ckpt", type=str, default=None,
                        help=f"Checkpoint path (default: {CKPT_DIR}/G_best.pth)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "single"],
                        help="Generate grid (default) or single image")
    parser.add_argument("--n", type=int, default=64,
                        help="Number of images in grid (must be perfect square, default: 64)")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(CKPT_DIR, "G_best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(CKPT_DIR, "G_final.pth")
            if not os.path.exists(ckpt_path):
                ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("G_epoch_")])
                if ckpts:
                    ckpt_path = os.path.join(CKPT_DIR, ckpts[-1])
                print(f"Using latest checkpoint: {ckpt_path}")
            else:
                print("No checkpoint found! Train first: py -3.11 train.py")
                exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    netG = load_generator(ckpt_path)

    output_path = args.output
    if output_path is None:
        os.makedirs(TEST_DIR, exist_ok=True)
        if args.mode == "grid":
            output_path = os.path.join(TEST_DIR, "test_grid.png")
        else:
            output_path = os.path.join(TEST_DIR, "test_single.png")

    if args.mode == "grid":
        generate_grid(netG, n=args.n, save_path=output_path)
    else:
        generate_single(netG, save_path=output_path)
