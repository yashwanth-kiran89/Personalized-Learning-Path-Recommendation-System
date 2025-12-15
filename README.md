
🎓 Personalized Learning Path Recommendation System
📌 Project Overview

This project implements a Personalized Learning Path Recommendation System that analyzes student behavior and learning patterns to recommend the most suitable learning path.

The goal of this project is not to chase high accuracy, but to demonstrate a complete, professional Machine Learning pipeline similar to real-world educational recommendation systems.

The system:

Processes raw student data

Learns learning-style patterns

Trains a neural network model

Produces stable, deterministic recommendations

Evaluates model behavior using proper ML metrics

🎯 What Problem Does This Solve?

Different students learn differently.

This system answers:

“Given a student’s learning style, progress, performance, and engagement — what learning path best suits them?”

🧠 Key Concepts Demonstrated

Data Preprocessing & Feature Engineering

Learning Style Identification

Recommendation Logic

Neural Network Training

Model Evaluation & Stability

End-to-End ML Pipeline Design

This project focuses on correct design and data flow, not shortcuts.

📊 Dataset Description (sample_data.csv)

Each row represents one student.

Column	Description
StudentID	Unique student identifier
Age	Student age
LearningStyle	Visual / Auditory / Kinesthetic / Reading-Writing
Progress	Learning progress (0–1)
CompletedCourses	Number of completed courses
AverageScore	Academic performance
PreferredContent	Videos / Articles / Quizzes / Podcasts
SessionTime	Average learning session time
🏗️ Project Structure
Personalized Learning Path Recommendation/
│
├── Data-Preprocessing.py
├── Learning-Style-Classification.py
├── Recommendation-Algorithm.py
├── Adaptive-Learning-System.py
├── Evaluation.py
│
├── sample_data.csv
├── preprocessed_student_data.csv
│
├── models/
│   ├── kmeans_learning_style.pkl
│   ├── decision_tree_learning_style.pkl
│   ├── adaptive_learning_model.h5
│   └── scaler.pkl
│
├── README.md
├── requirements.txt
└── venv/
