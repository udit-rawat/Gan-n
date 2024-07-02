# src/gan_training.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


latent_dim = 100


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_gan(X_train_minority, X_smoteen_minority, input_dim, epochs=1000, batch_size=32):
    generator = Generator(input_dim)
    discriminator = Discriminator(input_dim)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    adversarial_loss = nn.BCELoss()

    X_train_minority_tensor = torch.tensor(
        X_train_minority.values, dtype=torch.float32)
    X_smoteen_minority_tensor = torch.tensor(
        X_smoteen_minority.values, dtype=torch.float32)

    for epoch in range(epochs):
        idx = np.random.randint(
            0, X_train_minority_tensor.shape[0], batch_size)
        real_samples = X_train_minority_tensor[idx]
        real_labels = torch.ones((batch_size, 1), dtype=torch.float32)

        noise = torch.randn(batch_size, latent_dim)
        synthetic_samples = generator(noise)
        synthetic_labels = torch.zeros((batch_size, 1), dtype=torch.float32)

        optimizer_D.zero_grad()
        d_loss_real = adversarial_loss(
            discriminator(real_samples), real_labels)
        d_loss_synthetic = adversarial_loss(discriminator(
            synthetic_samples.detach()), synthetic_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_synthetic)
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        g_loss = adversarial_loss(
            discriminator(synthetic_samples), real_labels)
        g_loss.backward()
        optimizer_G.step()

        if epoch % 100 == 0:
            print(
                f"{epoch} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Add logging to check shapes
        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            print(f"Real samples shape: {real_samples.shape}")
            print(f"Noise shape: {noise.shape}")
            print(f"Synthetic samples shape: {synthetic_samples.shape}")
            print(f"Real labels shape: {real_labels.shape}")
            print(f"Synthetic labels shape: {synthetic_labels.shape}")

    torch.save(generator.state_dict(), "models/weights/gan.pth")

    return generator
