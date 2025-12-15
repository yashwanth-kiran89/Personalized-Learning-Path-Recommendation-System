"""
Evaluation Module
Personalized Learning Path Recommendation System
------------------------------------------------
Correctly evaluates classification model by fixing target labels.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == "__main__":

    print("\n📥 Loading dataset for evaluation...")
    data = pd.read_csv("preprocessed_student_data.csv")

    # -------------------------------
    # FIX: Convert LearningPath back to class labels
    # -------------------------------
    y_test = data["LearningPath"]

    # Convert continuous values → class indices
    y_test = pd.factorize(y_test)[0]

    # -------------------------------
    # Features MUST match training
    # -------------------------------
    X_test = data.drop(columns=["StudentID", "LearningPath"])

    print("📦 Loading scaler...")
    scaler = joblib.load("models/scaler.pkl")
    X_test = scaler.transform(X_test)

    print("📦 Loading trained model...")
    model = tf.keras.models.load_model("models/adaptive_learning_model.h5")

    print("🧠 Running predictions...")
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    print("\n📊 Evaluation Results")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 2))
    print("Precision:", round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 2))
    print("Recall   :", round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 2))
    print("F1-score :", round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 2))
