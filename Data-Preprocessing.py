"""
Data Preprocessing Module
Personalized Learning Path Recommendation System
------------------------------------------------

ROLE OF THIS FILE:
- Converts raw student data into ML-ready numerical format
- Handles missing values
- Encodes categorical attributes
- Scales numerical features
- Saves a clean dataset for downstream models

IMPORTANT:
This file is executed ONCE before model training.
It prepares the foundation for all ML components.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessing:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}

    def load_data(self, filepath):
        """Load raw student dataset"""
        return pd.read_csv(filepath)

    def handle_missing_values(self, data):
        """
        Handle missing values:
        - Numerical columns → mean
        - Categorical columns → mode
        """
        for col in data.columns:
            if data[col].dtype == "object":
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].mean(), inplace=True)
        return data

    def encode_categorical_features(self, data):
        """
        Convert categorical columns into numeric format
        using Label Encoding.
        """
        for col in data.columns:
            if data[col].dtype == "object":
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                self.encoders[col] = encoder
        return data

    def scale_numerical_features(self, data):
        """
        Apply feature scaling so that all numeric features
        contribute equally to model learning.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        return data

    def preprocess(self, filepath):
        """
        Complete preprocessing pipeline
        """
        print("Loading raw data...")
        data = self.load_data(filepath)

        print("Handling missing values...")
        data = self.handle_missing_values(data)

        print("Encoding categorical features...")
        data = self.encode_categorical_features(data)

        print("Scaling numerical features...")
        data = self.scale_numerical_features(data)

        return data


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":

    RAW_DATA_PATH = "sample_data.csv"
    OUTPUT_PATH = "preprocessed_student_data.csv"

    processor = DataPreprocessing()

    processed_data = processor.preprocess(RAW_DATA_PATH)

    processed_data.to_csv(OUTPUT_PATH, index=False)

    print("\nPreprocessing completed successfully.")
    print(f"Processed data saved to: {OUTPUT_PATH}")
