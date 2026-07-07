import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    DEVICE, DATA_ROOT, SAMPLES_DIR, CKPT_DIR, LOGS_DIR,
    Z_DIM, G_FEATURES, D_FEATURES, CHANNELS,
    BATCH_SIZE, EPOCHS, G_LR, D_LR, BETA1, BETA2,
    SAMPLE_INTERVAL, CKPT_INTERVAL,
)
from model import Generator, Discriminator, count_params
from dataset import get_dataloader
from utils import save_samples, plot_losses


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    loader = get_dataloader(DATA_ROOT, BATCH_SIZE)
    fixed_z = torch.randn(16, Z_DIM, device=DEVICE, dtype=torch.float32)

    netG = Generator(Z_DIM, G_FEATURES, CHANNELS).to(DEVICE)
    netD = Discriminator(D_FEATURES, CHANNELS).to(DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)

    print(f"Generator params:     {count_params(netG):,}")
    print(f"Discriminator params: {count_params(netD):,}")
    print(f"Device: {DEVICE}")
    print(f"Total batches/epoch: {len(loader)}")

    criterion = nn.MSELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=G_LR, betas=(BETA1, BETA2))
    optimizerD = optim.Adam(netD.parameters(), lr=D_LR, betas=(BETA1, BETA2))

    g_losses = []
    d_losses = []
    best_g_loss = float("inf")

    log_path = os.path.join(LOGS_DIR, "losses.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "g_loss", "d_loss"])

    for epoch in range(1, EPOCHS + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{EPOCHS}", leave=False)
        for i, (real_imgs, _) in enumerate(pbar):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)
            real_label = torch.full((batch_size, 1), 1.0, device=DEVICE, dtype=torch.float32)
            fake_label = torch.full((batch_size, 1), 0.0, device=DEVICE, dtype=torch.float32)

            netD.zero_grad()
            output_real = netD(real_imgs)
            lossD_real = criterion(output_real, real_label)

            noise = torch.randn(batch_size, Z_DIM, device=DEVICE, dtype=torch.float32)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            lossD_fake = criterion(output_fake, fake_label)

            lossD = (lossD_real + lossD_fake) * 0.5
            lossD.backward()
            optimizerD.step()

            netG.zero_grad()
            output = netD(fake)
            lossG = criterion(output, real_label)
            lossG.backward()
            optimizerG.step()

            epoch_g_loss += lossG.item()
            epoch_d_loss += lossD.item()
            n_batches += 1
            pbar.set_postfix(G=lossG.item(), D=lossD.item())

        avg_g_loss = epoch_g_loss / n_batches
        avg_d_loss = epoch_d_loss / n_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"Epoch [{epoch:3d}/{EPOCHS}]  G_loss: {avg_g_loss:.4f}  D_loss: {avg_d_loss:.4f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_g_loss:.4f}", f"{avg_d_loss:.4f}"])

        if epoch % SAMPLE_INTERVAL == 0:
            save_samples(netG, fixed_z, epoch, SAMPLES_DIR, DEVICE)

        if epoch % CKPT_INTERVAL == 0:
            torch.save(netG.state_dict(), os.path.join(CKPT_DIR, f"G_epoch_{epoch:03d}.pth"))
            torch.save(netD.state_dict(), os.path.join(CKPT_DIR, f"D_epoch_{epoch:03d}.pth"))

        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            torch.save(netG.state_dict(), os.path.join(CKPT_DIR, "G_best.pth"))
            torch.save(netD.state_dict(), os.path.join(CKPT_DIR, "D_best.pth"))

    torch.save(netG.state_dict(), os.path.join(CKPT_DIR, "G_final.pth"))
    torch.save(netD.state_dict(), os.path.join(CKPT_DIR, "D_final.pth"))
    plot_losses(g_losses, d_losses, LOGS_DIR)
    print("Training complete!")


if __name__ == "__main__":
    train()
