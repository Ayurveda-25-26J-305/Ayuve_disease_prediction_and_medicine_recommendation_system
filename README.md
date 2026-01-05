# Ayurvedic Herb Recommendation System

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A machine learning system that recommends Ayurvedic herbs based on patient profiles including disease, severity, dominant dosha, and age group.

## 📊 Quick Stats

-  **Top-1 Accuracy**: 52%
-  **Top-3 Accuracy**: 98% (SVM)
-  **Dataset Size**: 2,000 records
-  **Herb Classes**: 15 different Ayurvedic herbs
-  **Models Implemented**: 3 (Random Forest, Decision Tree, SVM)

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [File Structure](#file-structure)
- [Running in Google Colab](#running-in-google-colab)
- [Interpretation Guide](#interpretation-guide)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Clinical Use Disclaimer](#clinical-use-disclaimer)
- [Contributing](#contributing)
- [License](#license)

##  Overview

This project implements and compares three different machine learning algorithms to predict the most suitable Ayurvedic herbs for various health conditions:

-  **Random Forest Classifier**
-  **Decision Tree Classifier**
-  **Support Vector Machine (SVM)** ⭐ *Recommended*

  ✨ Key Features

✅ Multi-class classification for 15 different herbs
✅ Top-K prediction support (Top-1, Top-3, Top-5)
✅ Confidence scores for each recommendation
✅ Cross-validation for robust performance estimation
✅ Easy-to-use prediction pipeline
✅ Ready for deployment (.pkl model files)

📁 Dataset

Size: 2,000 patient records
Features: 4 categorical variables

Disease type
Severity level (Low, Medium, High)
Dominant Dosha (Vata, Pitta, Kapha)
Age group (Child, Adult, Elder)
