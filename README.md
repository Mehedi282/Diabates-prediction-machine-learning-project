# Diabates-prediction-machine-learning-project

[Diabetes prediction](https://colab.research.google.com/drive/19uCzO2EFhi0-TLLR0l3j7GIs3XqGIhHn)

This repository contains a machine learning project that predicts whether a person has diabetes based on medical diagnostic measurements. The project uses a Support Vector Machine (SVM) algorithm for classification.

Table of Contents
Introduction
Dataset
Usage
Model
Results


Introduction
The goal of this project is to predict whether a person has diabetes based on certain medical attributes. This project uses the Support Vector Machine (SVM) algorithm along with Standard Scaling for data normalization. The project also includes accuracy evaluation using the accuracy score metric from the scikit-learn library.

Dataset
The dataset used in this project is the PIMA Indians Diabetes Database, which is available on Kaggle or the UCI Machine Learning Repository. It consists of 768 samples, each containing 8 medical attributes (such as glucose level, insulin, BMI, etc.) and a target label indicating whether the individual has diabetes.

The script will:

Load and preprocess the dataset using pandas and numpy.
Split the data into training and testing sets.
Standardize the data using StandardScaler.
Train an SVM model using the scikit-learn library.
Evaluate the model's performance on the test data.


Model
This project uses a Support Vector Machine (SVM) for classification. The key steps are:

Data Scaling: Medical data often have varying ranges, so the data is standardized using StandardScaler.
Data Splitting: The dataset is split into training and testing sets using train_test_split.
Model Training: The SVM model is trained on the training data.
Model Evaluation: The model’s performance is measured using accuracy score on the test data.
Here are the key libraries used:

pandas for data manipulation.
numpy for numerical operations.
scikit-learn for machine learning algorithms (SVM, train_test_split, StandardScaler, and accuracy_score).
