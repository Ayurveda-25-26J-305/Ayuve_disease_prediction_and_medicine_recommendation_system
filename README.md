# Constitutional-Aware Ayurvedic Disease Prediction System

AI-powered disease prediction system integrating traditional Ayurvedic principles with modern machine learning.

## 🎯 Project Overview

This research project develops a machine learning system that predicts diseases based on symptoms while considering individual constitutional types (Prakriti) according to Ayurvedic medicine.

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

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

#### Using Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/Ayurvedic_Disease_Prediction_Training.ipynb
```

#### Using Python Script
```bash
python src/train_ayurvedic_model_UPDATED.py
```

### Making Predictions
```python
from predict_ayurvedic import AyurvedicPredictor

predictor = AyurvedicPredictor()
result = predictor.predict_disease(
    age=45, gender='Male',
    symptom='Joint pain', severity='Severe',
    duration_days=30,
    vata_score=0.6, pitta_score=0.2, kapha_score=0.2,
    prakriti='Vata'
)
```

## 📈 Results

| Metric | Constitutional-Aware | Basic (Symptom-Only) | Improvement |
|--------|---------------------|---------------------|-------------|
| Accuracy | 84.2% | 76.8% | +9.6% |
| Precision | 83.5% | 75.2% | +11.0% |
| Recall | 84.2% | 76.8% | +9.6% |
| F1-Score | 83.8% | 75.9% | +10.4% |

## 🎓 Academic Details

**Student:** Perera S I A  
**Registration:** IT22905918  
**Project ID:** 25-26J-305  
**Institution:** SLIIT - Sri Lanka Institute of Information Technology  
**Specialization:** Information Technology (IT)  
**Research Group:** CoEAI - Centre of Excellence for AI  

## 📚 Documentation

- [Complete Documentation](docs/Progress_Presentation_1_Documentation.md)
- [Dataset Summary](docs/UPDATED_DATASET_SUMMARY.md)
- [Training Guide](docs/HOW_TO_TRAIN.md)
- [Quick Reference](docs/Quick_Reference_Guide.md)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning:** scikit-learn
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Notebook:** Jupyter

## 🤝 Contributing

This is an academic research project. For questions or collaborations, please contact via university channels.

## 📄 License

This project is part of academic research at SLIIT.

## 🙏 Acknowledgments

- Classical Ayurvedic texts: Charaka Samhita, Sushruta Samhita, Madhava Nidana
- Dataset sources: Nawinna Ayurveda Hospital, Rajagiriya Ayurveda Center
- Supervisor: [Name]
- Co-Supervisor: [Name]

## 📧 Contact

**Student:** Perera S I A  
**Email:** [your-email]  
**Project:** Smart Ayurvedic Disease Prediction and Recommendation System

---

⭐ Star this repo if you find it useful!