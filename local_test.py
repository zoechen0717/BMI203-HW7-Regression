import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
from regression import BaseRegressor, LogisticRegressor, loadDataset

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

# Tests
def test_prediction():
    model.train_model(X_train, y_train, X_test, y_test)
    y_pred = model.make_prediction(X_test)
    assert np.all(y_pred >= 0) and np.all(y_pred <= 1), "Predictions should be between 0 and 1."

def test_loss_function():
    model.train_model(X_train, y_train, X_test, y_test)
    y_pred = model.make_prediction(X_test)
    sklearn_loss = log_loss(y_test, y_pred)
    model_loss = model.loss_function(y_test, y_pred)
    assert np.isclose(model_loss, sklearn_loss, atol=1e-4), "Loss function does not match sklearn log_loss."

def test_gradient():
    model.train_model(X_train, y_train, X_test, y_test)
    X_sample = X_train[:5]
    y_sample = y_train[:5]
    grad = model.calculate_gradient(y_sample, X_sample)
    assert grad.shape == (X_train.shape[1] + 1,), "Gradient shape mismatch."

def test_training():
    initial_weights = model.W.copy()
    model.train_model(X_train, y_train, X_test, y_test)
    updated_weights = model.W.copy()
    assert not np.array_equal(initial_weights, updated_weights), "Weights should update during training."

def print_confusion_matrix():
    model.train_model(X_train, y_train, X_test, y_test)
    y_pred = model.make_prediction(X_test)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    print_confusion_matrix()
