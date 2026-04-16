import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.feature_extraction import extract_features
from src.train_model import split_data, get_models, train_models
from src.model_evaluation import evaluate_models


# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Fix columns
df.columns = ['label', 'message', 'x1', 'x2', 'x3']
df = df[['label', 'message']]

print("\nDATA CHECK:")
print(df.head())

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# SIMPLE preprocessing (no removal)
df['cleaned'] = df['message'].astype(str)

# Remove empty rows
df = df[df['cleaned'].str.strip() != ""]

print("\nAfter cleaning:", df.shape)

# 🚨 CRITICAL CHECK
if df.shape[0] == 0:
    raise ValueError("Dataset is empty after cleaning!")

# Feature extraction
X, vectorizer = extract_features(df['cleaned'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Train
models = get_models()
trained_models = train_models(models, X_train, y_train)

# Evaluate
evaluate_models(trained_models, X_test, y_test)


import joblib
import os

# create folder if not exists
os.makedirs("models", exist_ok=True)

# Save ONE model (example: Naive Bayes)
joblib.dump(trained_models["Naive Bayes"], "models/spam_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved successfully!")