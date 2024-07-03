import xgboost as xgb
from sklearn.metrics import classification_report
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split


def train_xgboost(X_train, y_train, X_test, y_test, output_file):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_file, index=False)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Model")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    augmented_data = pd.read_csv(args.input)
    X_train = pd.read_csv(
        '/Users/uditrawat/Desktop/RL/GAN-f/data/augmented/X_augmented.csv')
    y_train = pd.read_csv(
        '/Users/uditrawat/Desktop/RL/GAN-f/data/augmented/y_augmented.csv')
    y_test = pd.read_csv(
        '/Users/uditrawat/Desktop/RL/GAN-f/data/processed/y_test.csv')
    X_test = pd.read_csv(
        '/Users/uditrawat/Desktop/RL/GAN-f/data/processed/X_test.csv')
    train_xgboost(X_train, y_train, X_test, y_test, args.output)
