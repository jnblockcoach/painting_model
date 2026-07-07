import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, g_features=512, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_features, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_features, g_features // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_features // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_features // 2, g_features // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_features // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_features // 4, g_features // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_features // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_features // 8, channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, d_features=64, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, d_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_features, d_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_features * 2, d_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_features * 4, d_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_features * 8, 1, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from config import Z_DIM, G_FEATURES, D_FEATURES, CHANNELS
    g = Generator(Z_DIM, G_FEATURES, CHANNELS)
    d = Discriminator(D_FEATURES, CHANNELS)
    print(f"Generator params:     {count_params(g):,}")
    print(f"Discriminator params: {count_params(d):,}")
    print(f"Total params:         {count_params(g) + count_params(d):,}")

    test_z = torch.randn(4, Z_DIM)
    test_img = g(test_z)
    test_out = d(test_img)
    print(f"Input z shape:  {test_z.shape}")
    print(f"Output img shape: {test_img.shape}")
    print(f"Discriminator out:  {test_out.shape}")
