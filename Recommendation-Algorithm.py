"""
Recommendation Algorithm Module
Personalized Learning Path Recommendation System
------------------------------------------------
Generates human-readable learning path recommendations
using the trained learning style classifier.
"""

import pandas as pd
import joblib
import os

# -------------------- LEARNING PATH MAP --------------------

LEARNING_PATH_MAP = {
    0: "Beginner – Concept-Focused Learning Path",
    1: "Intermediate – Practice-Oriented Learning Path",
    2: "Advanced – Project-Based Learning Path",
    3: "Mixed – Adaptive Multimedia Learning Path"
}

# -------------------- CLASS --------------------

class RecommendationSystem:
    def __init__(self):
        print("Loading trained learning style classifier...")
        self.classifier = joblib.load("models/learning_style_classifier.pkl")

    def load_data(self, filepath):
        return pd.read_csv(filepath)

    def recommend(self, student_row):
        """
        Predict learning style and map it to a learning path.
        """
        predicted_style = self.classifier.predict(student_row)[0]
        recommended_path = LEARNING_PATH_MAP.get(
            predicted_style, "General Learning Path"
        )
        return predicted_style, recommended_path


# -------------------- MAIN --------------------

if __name__ == "__main__":

    DATA_PATH = "preprocessed_student_data.csv"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "❌ Preprocessed data not found. Run Data-Preprocessing.py first."
        )

    recommender = RecommendationSystem()
    data = recommender.load_data(DATA_PATH)

    print("\n📊 Sample of Preprocessed Student Data:")
    print(data.head())

    # -------------------- SELECT ONE STUDENT --------------------
    # IMPORTANT: DO NOT DROP StudentID (model was trained with it)
    sample_student = data.iloc[[0]]

    style_id, path = recommender.recommend(sample_student)

    print("\n🎯 Recommendation Result")
    print("------------------------")
    print(f"Predicted Learning Style ID : {style_id}")
    print(f"Recommended Learning Path   : {path}")
