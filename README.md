Personalized Ayurvedic Meal Planner

A web-based AI-powered Ayurvedic Meal Recommendation System that provides personalized dietary guidance based on user inputs. The system recommends a well-balanced meal, calculates nutrient values, suggests foods to avoid, and identifies the dominant dosha according to Ayurvedic principles.

📝Features

Personalized Meal Plans: Input your age, gender, disease, meal category, food preference, and activity level to receive a customized Ayurvedic meal plan.

Nutrient Summary: Calories, protein, carbohydrates, and fat are calculated for the recommended meal.

Foods to Avoid: Shows foods that should be limited or avoided based on disease and user inputs.

Dosha Identification: Determines the dominant dosha (Vata, Pitta, Kapha) for the user.

Frontend & Backend Integration: React frontend with FastAPI backend for real-time responses.

Future Enhancement: Option to generate daily/weekly/monthly health reports.

⚙️ Tech Stack

Frontend:

React.js

CSS (Deep Forest Green, White, Black theme)

Vite as build tool

Backend:

Python FastAPI

scikit-learn (Random Forest)

pandas, numpy

🧪 Model Training

The system uses a Random Forest Classifier for multi-output prediction of meals.

Metrics tracked: F1 Score, Accuracy, Precision, Recall.

Current performance (on test set):

F1 Score: 0.80

Accuracy, Precision, Recall: calculated per run using train/test split.

The model is trained on encoded features like age, gender, disease, meal category, food preference, and activity level.

