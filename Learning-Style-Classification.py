"""
Learning Style Classification Module
Personalized Learning Path Recommendation System
------------------------------------------------

ROLE OF THIS FILE:
- Discover natural learning behavior patterns using clustering
- Train a supervised model to predict learning style for new students
- Persist trained models for downstream recommendation usage
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib


class LearningStyleClassification:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.classifier = DecisionTreeClassifier(random_state=42)
        self.encoders = {}

    def load_data(self, filepath):
        """Load preprocessed student data"""
        return pd.read_csv(filepath)

    def encode_categorical_features(self, data):
        """Encode categorical features for ML models"""
        for col in data.columns:
            if data[col].dtype == "object":
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                self.encoders[col] = encoder
        return data

    def cluster_students(self, data):
        """
        Discover learning-style clusters using K-Means
        """
        features = data.drop(columns=["StudentID"], errors="ignore")

        print("Clustering students using K-Means...")
        data["DiscoveredLearningStyle"] = self.kmeans.fit_predict(features)

        return data

    def train_classifier(self, data):
        """
        Train Decision Tree to predict discovered learning styles
        """
        X = data.drop(columns=["DiscoveredLearningStyle"])
        y = data["DiscoveredLearningStyle"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("\nTraining Decision Tree classifier...")
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)

        print("\nClassification Results:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0  # avoids noisy warnings for small datasets
            )
        )

    def save_models(self):
        """
        Save trained models safely
        """
        os.makedirs("models", exist_ok=True)

        joblib.dump(self.kmeans, "models/kmeans_learning_style.pkl")
        joblib.dump(self.classifier, "models/learning_style_classifier.pkl")

        print("\nModels saved successfully in 'models/' directory.")

    def load_models(self):
        """Load trained models"""
        self.kmeans = joblib.load("models/kmeans_learning_style.pkl")
        self.classifier = joblib.load("models/learning_style_classifier.pkl")

    def predict_learning_style(self, student_data):
        """Predict learning style for a new student"""
        return self.classifier.predict(student_data)[0]


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":

    DATA_PATH = "preprocessed_student_data.csv"

    system = LearningStyleClassification(n_clusters=4)

    data = system.load_data(DATA_PATH)
    data = system.encode_categorical_features(data)

    clustered_data = system.cluster_students(data)

    system.train_classifier(clustered_data)

    system.save_models()
