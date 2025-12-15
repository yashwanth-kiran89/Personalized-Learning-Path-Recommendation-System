"""
Adaptive Learning System
Personalized Learning Path Recommendation System
------------------------------------------------
End-to-end working ML pipeline with saved scaler and model.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class AdaptiveLearningSystem:
    def __init__(self, input_size, output_size):
        self.scaler = StandardScaler()
        self.model = self._build_model(input_size, output_size)

    def _build_model(self, input_size, output_size):
        model = Sequential([
            Input(shape=(input_size,)),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(output_size, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=8,
            verbose=1
        )

        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nModel Accuracy: {acc:.2f}")


if __name__ == "__main__":

    print("📥 Loading dataset...")
    data = pd.read_csv("preprocessed_student_data.csv")
    print(data.head())

    # -------------------------------
    # Separate features and target
    # -------------------------------
    X = data.drop(columns=["StudentID", "LearningPath"])
    y = data["LearningPath"]

    # -------------------------------
    # Encode categorical features
    # -------------------------------
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])

    y = LabelEncoder().fit_transform(y)

    # -------------------------------
    # Initialize system
    # -------------------------------
    system = AdaptiveLearningSystem(
        input_size=X.shape[1],
        output_size=len(np.unique(y))
    )

    # -------------------------------
    # Scale features & SAVE scaler
    # -------------------------------
    X_scaled = system.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(system.scaler, "models/scaler.pkl")

    # -------------------------------
    # Train model & SAVE model
    # -------------------------------
    system.train(X_scaled, y)
    system.model.save("models/adaptive_learning_model.h5")

    # -------------------------------
    # Predict for one student
    # -------------------------------
    sample_student = X.iloc[[0]]
    sample_student_scaled = system.transform(sample_student)
    prediction = system.model.predict(sample_student_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]

    print("\n🎯 Predicted learning path (encoded):", predicted_class)
