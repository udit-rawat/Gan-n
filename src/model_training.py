# src/model_training.py

import xgboost as xgb
from sklearn.metrics import classification_report


def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model
