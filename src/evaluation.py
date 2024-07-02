# src/evaluation.py
import numpy as np
import torch


def augment_data(X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator, latent_dim=100):
    num_samples_to_generate = len(X_train_minority)
    noise = torch.randn(num_samples_to_generate, latent_dim)
    generated_samples = generator(noise).detach().numpy()

    X_augmented = np.vstack((X_train_minority, generated_samples))
    y_augmented = np.hstack(
        (y_train_minority, [y_train_minority.iloc[0]] * num_samples_to_generate))

    X_final = np.vstack((X_augmented, X_train_majority))
    y_final = np.hstack((y_augmented, y_train_majority))

    return X_final, y_final
