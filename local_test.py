import numpy as np
import pandas as pd
from regression import BaseRegressor, LogisticRegressor,loadDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Load dataset
features = [
    'Penicillin V Potassium 500 MG',
    'Computed tomography of chest and abdomen',
    'Plain chest X-ray (procedure)',
    'Low Density Lipoprotein Cholesterol',
    'Creatinine',
    'AGE_DIAGNOSIS'
]
# Load dataset for testing
X_train, X_test, y_train, y_test = loadDataset(features, split_percent=0.8)

# Initialize model
model = LogisticRegressor(num_feats=X_train.shape[1])

# Initialize our model
our_model = LogisticRegressor(num_feats=X_train.shape[1])
our_model.train_model(X_train, y_train, X_test, y_test)
y_pred_our = our_model.make_prediction(X_test)
y_pred_our_binary = (y_pred_our >= 0.5).astype(int)

# Initialize sklearn model
sklearn_model = LogisticRegression()
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict_proba(X_test)[:, 1]
sklearn_pred_binary = sklearn_model.predict(X_test)

# Compare accuracy and log loss
our_accuracy = accuracy_score(y_test, y_pred_our_binary)
sklearn_accuracy = accuracy_score(y_test, sklearn_pred_binary)
our_log_loss = log_loss(y_test, y_pred_our)
sklearn_log_loss = log_loss(y_test, sklearn_pred)

print(f"Our Model Accuracy: {our_accuracy:.4f}")
print(f"Sklearn Model Accuracy: {sklearn_accuracy:.4f}")
print(f"Our Model Log Loss: {our_log_loss:.4f}")
print(f"Sklearn Model Log Loss: {sklearn_log_loss:.4f}")

assert np.allclose(our_log_loss, sklearn_log_loss, atol=0.1), "Log loss mismatch with sklearn!"
assert np.abs(our_accuracy - sklearn_accuracy) < 0.1, "Accuracy mismatch with sklearn!"
