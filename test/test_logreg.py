"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from regression import BaseRegressor, LogisticRegressor,loadDataset
# (you will probably need to import more things here)


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

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_test)
# Initialize model
model = LogisticRegressor(num_feats=6)

# Tests
def test_prediction():
    # Train model
    model.train_model(X_train, y_train, X_test, y_test)
    # Make predictions
    y_pred = model.make_prediction(X_test)
    # Check that predictions are between 0 and 1
    assert np.all(y_pred >= 0) and np.all(y_pred <= 1), "Predictions should be between 0 and 1."

# Test that the loss function is being calculated correctly
def test_loss_function():
    # Train model
    model.train_model(X_train, y_train, X_test, y_test)
    # Make predictions
    y_pred = model.make_prediction(X_test)
    # Calculate loss
    sklearn_loss = log_loss(y_test, y_pred)
    # Calculate loss using model
    model_loss = model.loss_function(y_test, y_pred)
    # Check that the loss functions match
    assert np.isclose(model_loss, sklearn_loss, atol=1e-3), "Loss function does not match sklearn log_loss."

# Test that the gradient is being calculated correctly
def test_gradient():
    # Train model
    model.train_model(X_train, y_train, X_test, y_test)
    # Calculate gradient
    X_sample = X_train[:5]
    y_sample = y_train[:5]
    # Calculate gradient
    grad = model.calculate_gradient(y_sample, X_sample)
    # Check that the gradient shape is correct
    assert grad.shape == (X_train.shape[1] + 1,), "Gradient shape mismatch."

# Test that weights update during training
def test_training():
    # Get initial weights
    initial_weights = model.W.copy()
    # Train model
    model.train_model(X_train, y_train, X_test, y_test)
    # Get updated weights
    updated_weights = model.W.copy()
    # Check that weights have updated
    assert not np.array_equal(initial_weights, updated_weights), "Weights should update during training."

    