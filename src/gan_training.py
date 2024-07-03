import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import pickle

latent_dim = 100


class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim):
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
    generator = Generator(input_dim, latent_dim)
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

    return generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with preprocessed data")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the generator model")
    args = parser.parse_args()

    X_train = pd.read_csv(f"{args.input_dir}/X_train.csv")
    X_resampled = pd.read_csv(f"{args.input_dir}/X_resampled.csv")
    y_train = pd.read_csv(f"{args.input_dir}/y_train.csv")
    y_resampled = pd.read_csv(f"{args.input_dir}/y_resampled.csv")

    minority_class = y_train['Class'].value_counts().idxmin()
    X_train_minority = X_train[y_train['Class'] == minority_class]
    X_smoteen_minority = X_resampled[y_resampled['Class'] == minority_class]

    generator = train_gan(X_train_minority, X_smoteen_minority,
                          input_dim=X_train.shape[1], epochs=1000, batch_size=32)

    # Save the entire generator object
    with open(args.output, 'wb') as f:
        pickle.dump(generator, f)
