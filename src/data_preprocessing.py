# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from imblearn.combine import SMOTEENN
from sklearn.impute import KNNImputer


def load_and_preprocess_data(target, file):
    # Fetch dataset
    df = pd.read_csv(file)
    X = df.drop(columns=[target])
    y = df[target]
    y = y.map({2: 0, 4: 1})

    # Initialize the KNN Imputer with 5 neighbors
    imputer = KNNImputer(n_neighbors=5)

    # Impute missing values in X
    X_imputed = imputer.fit_transform(X)

    # Replace original X with imputed data
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Apply SMOTEENN
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, X_resampled, y_resampled
