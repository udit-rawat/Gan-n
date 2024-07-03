import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Ensure this is correctly imported
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_xgboost
from src.augment_data import augment_data
from src.gan_training import train_gan


def main(target, file):
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, X_resampled, y_resampled = load_and_preprocess_data(
        target, file)

    print("Data preprocessing completed.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    # Separate minority and majority classes
    minority_class = y_train.value_counts().idxmin()
    X_train_minority = X_train[y_train == minority_class]
    y_train_minority = y_train[y_train == minority_class]
    X_train_majority = X_train[y_train != minority_class]
    y_train_majority = y_train[y_train != minority_class]

    # Extract SMOTEEN-ed minority class
    X_smoteen_minority = X_resampled[y_resampled == minority_class]
    print("Data preprocessing completed.")
    print(f"X_smoteen_minority shape: {X_smoteen_minority.shape}")
    print(f"X_train_minority shape: {X_train_minority.shape}")
    print("Extraction completed.")
    print("Training GAN...")
    generator = train_gan(
        X_train_minority, X_smoteen_minority, input_dim=9, epochs=10, batch_size=32)

    X_final, y_final = augment_data(
        X_train_minority, X_train_majority, y_train_minority, y_train_majority, generator)
    train_xgboost(X_final, y_final, X_test, y_test)


if __name__ == "__main__":
    target = 'Class'  # You can change this ID to fetch a different dataset
    file = "/Users/uditrawat/Desktop/RL/GAN-f/dataset/15_csv.csv"
    main(target=target, file=file)
