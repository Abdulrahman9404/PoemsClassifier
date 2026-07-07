# Arabic Poetry Meter Classification

This project explores the use of machine learning for classifying Arabic poetry by its poetic meter (البحر الشعري). It implements a complete text classification pipeline, including data preprocessing, feature extraction, model training, and evaluation.

## Features

* Loads and preprocesses an Arabic poetry dataset
* Optional removal of Arabic diacritics (Harakat)
* Extracts character n-gram features using TF-IDF
* Trains a Logistic Regression classifier
* Evaluates performance using Accuracy, Macro F1-score, and a Classification Report

## Technologies

* Python
* pandas
* scikit-learn

## Model

* **Feature Extraction:** TF-IDF Character n-grams (2–5)
* **Classifier:** Logistic Regression

## Results

The primary goal of this project was to gain hands-on experience with natural language processing and text classification in Python. It demonstrates the complete workflow of preparing text data, training a machine learning model, and evaluating its performance.

## Running the Project

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Place the dataset in the `data/` directory.

3. Run the classifier:

```bash
python meter_classifier.py
```
