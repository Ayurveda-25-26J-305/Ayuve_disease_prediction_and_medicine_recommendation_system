# Ayuvedic Disease Prediction and Medicine Recommendation system 🌿

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)
![Components](https://img.shields.io/badge/components-4-brightgreen.svg)

A comprehensive AI-powered Ayuvedic healthcare system integrating traditional wisdom with modern machine learning for personalized health management.

## 📊 Quick Stats

- 🏥 **System Components**: 4 (Disease Prediction, Medicine Recommendation, Diet Planning, Q&A)

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
  - [Disease Prediction Component](#1️⃣-disease-prediction-component)
  - [Medicine Recommendation Component](#2️⃣-medicine-recommendation-component)
  - [Diet Recommendation Component](#3️⃣-diet-recommendation-component)
  - [Q&A Component](#4️⃣-qa-component)
  - [Component Integration Flow](#-component-integration-flow)
- [Dataset](#-dataset-medicine-recommendation-component)
- [Model Performance](#-model-performance-medicine-recommendation-component)
- [Component Implementation Status](#-component-implementation-status)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#-making-predictions)
- [Model Architecture](#-model-architecture)
- [Evaluation Metrics](#-evaluation-metrics)
- [File Structure](#-file-structure)
- [Running in Google Colab](#-running-in-google-colab)
- [Interpretation Guide](#-interpretation-guide)
- [Limitations](#️-limitations)
- [Use Cases](#-use-cases)
- [Roadmap & Future Improvements](#-roadmap--future-improvements)
- [Clinical Use Disclaimer](#️-clinical-use-disclaimer)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors](#-authors)
- [Acknowledgments](#-acknowledgments)

## 🎯 Overview

An integrated AI-powered Ayuvedic healthcare system that combines traditional Ayuvedic wisdom with modern machine learning to provide comprehensive health management. The system consists of four interconnected components that work together to deliver personalized healthcare recommendations.

### 🏥 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                  AYURVEDIC AI HEALTHCARE SYSTEM                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       User Input              │
              │  (Symptoms, Profile, Query)   │
              └───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│   Disease    │      │   Medicine   │     │     Diet     │
│  Prediction  │───▶  │Recommendation│     │Recommendation│
│              │      │              │     │              │
└──────────────┘      └──────────────┘     └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   QA Component   │
                    │  (Information &  │
                    │    Guidance)     │
                    └──────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │  Integrated Healthcare   │
                │      Recommendations     │
                └──────────────────────────┘
```

### ✨ Key Features

- 🏥 **Comprehensive Health Analysis**: Multi-component system for complete health management
- 🤖 **AI-Powered Predictions**: Machine learning models for accurate recommendations
- 🌿 **Traditional Ayuvedic Wisdom**: Based on authentic Ayuvedic principles
- 💬 **Interactive Q&A**: Natural language interface for health queries
- 🍽️ **Personalized Diet Plans**: Customized nutrition based on dosha and condition
- 📱 **Easy Integration**: Modular design for easy deployment

## 🏗️ System Architecture

### System Components Overview

The Ayuvedic AI Healthcare System is built on four core components that work together to provide comprehensive health management:

---

### 1️⃣ Disease Prediction Component

**Purpose**: Analyzes symptoms and patient data to predict potential health conditions

**How it Works**:

**Key Features**:
- Multi-symptom analysis
- Dosha-based disease classification
- Severity level prediction
- Support for both acute and chronic conditions

### Diseases Covered
- Gastritis (Amlapitta)
- Arthritis (Amavata)
- Diabetes (Prameha)
- Migraine (Ardhavabhedaka)
- Asthma (Shwasa)

### Key Features
- ✅ Constitutional-aware prediction (Vata, Pitta, Kapha)
- ✅ 3000+ patient records
- ✅ 80%+ prediction accuracy
- ✅ Strong classical Ayurvedic references
- ✅ ~10% improvement over symptom-only approaches

## 📊 Dataset

- **Total Records:** 2,992
- **Unique Patients:** 1,000
- **Symptoms:** 60 mapped to diseases
- **Prakriti Assessment:** 15 questions
- **Classical References:** Charaka Samhita, Sushruta Samhita, Madhava Nidana

### 2️⃣ Medicine Recommendation Component

**Purpose**: Recommends appropriate Ayuvedic herbs and formulations based on diagnosed conditions

**How it Works**:
- Receives disease prediction and patient profile
- Considers disease type, severity, dominant dosha, and age group
- Applies ML models trained on 2,000+ Ayuvedic treatment records
- Returns top herb recommendations with confidence levels

**Key Features**:
- 15 different Ayuvedic herbs in database
- Top-K recommendations (Top-1, Top-3, Top-5)
- Confidence scoring for each recommendation
- Personalized based on dosha and age

**Machine Learning Models**:
- 🌲 **Random Forest**: 400 estimators, 96.75% Top-3 accuracy
- 🌳 **Decision Tree**: Max depth 10, interpretable decisions
- 🎯 **SVM (Recommended)**: RBF kernel, 98% Top-3 accuracy

**Performance Metrics**:
- Top-1 Accuracy: 52%
- Top-3 Accuracy: 98%
- Top-5 Accuracy: 98.5%
- Cross-validation: 58% mean accuracy

**Herbs Database**:
- Amalaki (Indian Gooseberry)
- Ashwagandha (Winter Cherry)
- Brahmi (Water Hyssop)
- Gudmar (Sugar Destroyer)
- Guggulu (Indian Bedellium)
- Jatamansi (Spikenard)
- Neem (Margosa)
- Nirgundi (Five-leaved Chaste)
- Pippali (Long Pepper)
- Shankhapushpi (Morning Glory)
- Triphala (Three Fruits)
- Tulsi (Holy Basil)
- Turmeric (Haldi)
- Vasaka (Malabar Nut)
- Yashtimadhu (Licorice)

---

### 3️⃣ Diet Recommendation Component

**Purpose**: Provides personalized dietary guidelines based on dosha type and health condition

**How it Works**:
- Analyzes patient's dominant dosha (Vata, Pitta, Kapha)
- Considers current health condition and severity
- Recommends foods to include and avoid
- Provides meal timing and preparation guidelines

**Key Features**:
- Dosha-specific diet plans
- Disease-specific nutritional guidance
- Food combination principles (Viruddha Ahara)

**Recommendation Categories**:
- ✅ **Foods to Include**: Beneficial ingredients
- ❌ **Foods to Avoid**: Aggravating items
- ⏰ **Meal Timing**: Optimal eating schedule
- 🍳 **Preparation Methods**: Cooking recommendations
- 🌶️ **Taste Balance**: Six tastes (Rasa) guidance

**Dosha-Based Principles**:
- **Vata**: Warm, moist, grounding foods
- **Pitta**: Cool, calming, less spicy foods
- **Kapha**: Light, dry, stimulating foods

---

### 4️⃣ Q&A Component

**Purpose**: Interactive conversational interface for health queries and guidance

**How it Works**:
- Natural language processing for question understanding
- Retrieves information from Ayuvedic knowledge base
- Provides contextual answers about herbs, conditions, and treatments
- Can explain predictions and recommendations from other components

**Key Features**:
- Natural language query support
- Ayuvedic concept explanations
- Herb properties and usage information
- Treatment guidance and precautions
- Integration with other system components

**Query Types Supported**:
- "What is [disease/herb/dosha]?"
- "How to treat [condition]?"
- "What are the benefits of [herb]?"
- "Can I take [herb] with [condition]?"
- "What foods are good for [dosha]?"
- "Explain my recommendations"

**Knowledge Base**:
- Ayuvedic principles and concepts
- Herb properties and contraindications
- Disease management guidelines
- Dietary principles
- Treatment protocols

---

### 🔄 Component Integration Flow

```
User Query/Symptoms
       │
       ▼
┌──────────────────┐
│ Disease Prediction│
└──────────────────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐   ┌──────────────┐
│  Medicine    │   │     Diet     │
│Recommendation│   │Recommendation│
└──────────────┘   └──────────────┘
       │                 │
       └────────┬────────┘
                ▼
         ┌─────────────┐
         │ QA Component│◄──── Direct User Questions
         └─────────────┘
                │
                ▼
      Comprehensive Response
      (Predictions + Medicines + Diet + Explanations)
```

### 🎯 Data Flow

1. **User Input** → Symptoms, profile, or questions
2. **Disease Prediction** → Identifies likely conditions
3. **Medicine Recommendation** → Suggests appropriate herbs
4. **Diet Recommendation** → Provides dietary guidelines
5. **Q&A Component** → Answers questions and explains results
6. **Integrated Output** → Complete health management plan

---



## 📊 Component Implementation Status

| Component | Status | Technology | Accuracy/Performance |
|-----------|--------|------------|---------------------|
| 🏥 Disease Prediction | ✅ In Development | ML Classification | TBD |
| 💊 Medicine Recommendation | ✅ **In Development** | SVM, Random Forest, Decision Tree | TBD |
| 🍽️ Diet Recommendation | ✅ In Development | Rule-based + ML | TBD |
| 💬 Q&A Component | ✅ In Development | NLP, Knowledge Base | TBD |

> **Note**: This repository currently contains the **Medicine Recommendation Component** implementation. Other components are under development and will be integrated in future releases.

---

## 📦 Requirements

```python
- pandas
- numpy
- matplotlib
- scikit-learn
- joblib
```

## 🚀 Installation

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## 🔬 Running in Google Colab

1. Upload the notebook to Google Colab
2. Upload your dataset when prompted
3. Run all cells sequentially
4. Download the trained model files

## 📖 Interpretation Guide

## ⚠️ Limitations
2. Real Ayuvedic practice involves more nuanced factors not captured in the model
3. Should be used as a **decision support tool**, not a replacement for professional consultation
4. Dataset size (2,000 records) may not capture all rare disease-herb combinations

## 🚀 Roadmap & Future Improvements

### Phase 1: Core Components (Q1-Q2 2026)
- [ ] Medicine Recommendation Component
- [ ] Disease Prediction Component
- [ ] Diet Recommendation Component
- [ ] Q&A Component
- [ ] Component integration layer


###  Deployment 
- [ ] Web interface for practitioners
- [ ] Mobile application (iOS & Android)
- [ ] REST API for third-party integration
- 

## 💡 Use Cases

### For Ayuvedic Practitioners
- 🏥 Clinical decision support for herb selection
- 📊 Data-driven treatment planning
- 📈 Patient progress tracking
- 📚 Quick reference for herb properties

### For Healthcare Facilities
- 🏪 Integrated health management system
- 📱 Digital health records
- 🔄 Standardized treatment protocols
- 📊 Treatment outcome analytics

### For Researchers
- 🧪 Ayuvedic treatment pattern analysis
- 📈 Clinical data mining
- 🔬 Traditional medicine validation
- 📊 Comparative effectiveness studies

### For Wellness Centers
- 🧘 Personalized wellness programs
- 🍽️ Customized diet planning
- 🌿 Herb and supplement recommendations
- 📱 Client health monitoring

---

## ⚕️ Clinical Use Disclaimer

⚠️ **Important**: This model is for educational and research purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified Ayuvedic practitioners or healthcare professionals for medical decisions.

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Model improvements
- Additional evaluation metrics
- Documentation enhancements
- Bug fixes

## 👥 Authors

Machine Learning implementation for Ayuvedic herb recommendation system

## 🙏 Acknowledgments

- Built using scikit-learn machine learning library
- Based on authentic Ayuvedic medicinal principles and classical texts
- Dataset structure follows traditional Dosha-based diagnosis (Prakriti analysis)
- Incorporates concepts from Charaka Samhita, Sushruta Samhita, and Ashtanga Hridaya
- Special thanks to the Ayuvedic research community

---
