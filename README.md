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
-  ✅ Multi-class classification for 15 different herbs
-  ✅ Top-K prediction support (Top-1, Top-3, Top-5)
-  ✅ Confidence scores for each recommendation
-  ✅ Cross-validation for robust performance estimation
-  ✅ Easy-to-use prediction pipeline
-  ✅ Ready for deployment (.pkl model files)

## 📁 Dataset

- **Size**: 2,000 patient records
- **Features**: 4 categorical variables
  - Disease type
  - Severity level (Low, Medium, High)
  - Dominant Dosha (Vata, Pitta, Kapha)
  - Age group (Child, Adult, Elder)
- **Target**: Recommended herb (15 different Ayurvedic herbs)

### 🌿 Herbs in Dataset
Amalaki, Ashwagandha, Brahmi, Gudmar, Guggulu, Jatamansi, Neem, Nirgundi, Pippali, Shankhapushpi, Triphala, Tulsi, Turmeric, Vasaka, Yashtimadhu

## 🏆 Model Performance

### Summary Table

| Model | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | CV Mean | CV Std |
|-------|---------------|----------------|----------------|---------|--------|
| Random Forest | 52% | 96.75% | 98% | 57.55% | 1.95% |
| Decision Tree | 52% | 96.75% | 98% | 58% | 0.88% |
| SVM | 52% | 98% | 98.5% | 58% | 0.88% |



### 💡 Key Findings

- ✅ All three models achieve similar **Top-1 accuracy (~52%)**, indicating the complexity of the multi-class classification problem
- 🎯 **Top-3 accuracy** is excellent (>96%), meaning the correct herb is almost always in the top 3 recommendations
- ⭐ **SVM** shows slightly better Top-3 performance (98%) and is the recommended model for deployment
- 📊 Cross-validation results show consistent performance across all folds

## 📦 Requirements

```python
pandas
numpy
matplotlib
scikit-learn
joblib
```

## 🚀 Installation

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## 🏗️ Model Architecture

### Preprocessing Pipeline
- **One-Hot Encoding**: Converts categorical features into numerical format
- Handles unknown categories gracefully during prediction

### Random Forest Configuration
- **Estimators**: 400 trees
- **Random State**: 42 (for reproducibility)

### Decision Tree Configuration
- **Max Depth**: 10 (prevents overfitting)
- **Random State**: 42

### SVM Configuration
- **Kernel**: RBF (Radial Basis Function)
- **C**: 1.0 (regularization parameter)
- **Gamma**: scale
- **Probability**: True (enables probability estimates for Top-K predictions)

## 📈 Evaluation Metrics

### Top-K Accuracy
- **Top-1**: Exact match accuracy
- **Top-3**: Correct herb is in top 3 predictions
- **Top-5**: Correct herb is in top 5 predictions

### Cross-Validation
- **Method**: 5-fold Stratified K-Fold
- Ensures balanced class distribution in each fold
- Provides robust performance estimates

### Classification Report
Includes precision, recall, and F1-score for each herb class

🔬 Running in Google Colab

Upload the notebook to Google Colab
Upload your dataset when prompted
Run all cells sequentially
Download the trained model files

## 🚀 Future Improvements

- [ ] Expand dataset with more patient records
- [ ] Include additional features (symptoms, medical history, constitution subtypes)
- [ ] Implement ensemble methods combining multiple models
- [ ] Add contraindication checking
- [ ] Develop web/mobile interface for practitioners
- [ ] Include herb interaction warnings
- [ ] Multi-label classification for herb combinations

## ⚕️ Clinical Use Disclaimer

⚠️ **Important**: This model is for educational and research purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified Ayurvedic practitioners or healthcare professionals for medical decisions.

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Model improvements
- Additional evaluation metrics
- Documentation enhancements
- Bug fixes

## 👥 Authors

Machine Learning implementation for Ayurvedic herb recommendation system

## 🙏 Acknowledgments

- Built using scikit-learn machine learning library
- Based on Ayurvedic medicinal principles
- Dataset structure follows traditional Dosha-based diagnosis

---
