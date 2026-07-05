Name

Farmer Crop & Fertilizer Recommendation System

Description

This is a Machine Learning based web application built using Flask that helps farmers to:

Recommend the best crop based on soil and environmental conditions 🌱
Suggest the most suitable fertilizer based on soil, crop type, and nutrients 🧪

The system uses trained ML models (Random Forest) to provide accurate predictions along with meaningful remarks for better farming decisions.


Features
🌱 Crop Recommendation System
🧪 Fertilizer Recommendation System
📊 High accuracy ML model (Random Forest)
💬 Fertilizer explanation (Remark system)
🖥️ Simple and user-friendly web interface
📁 Flask-based backend integration
🎯 Real-time predictions

Technologies Used
Python 🐍
Flask 🌐
Machine Learning (Scikit-learn)
Pandas & NumPy
HTML, CSS
Joblib / Pickle

Farmer-crop-and-fertilizer/
│
├── app.py
├── model/
│   ├── crop_label_model.pkl
│   ├── crop_label_scaler.pkl
│   ├── crop_label_encoder.pkl
│   ├── fertilizer_model.pkl
│   ├── fertilizer_scaler.pkl
│   ├── fertilizer_encoder.pkl
│   ├── soil_encoder.pkl
│   ├── crop_encoder.pkl
│   ├── fertilizer_remark.pkl
│
├── templates/
│   ├── index.html
│   ├── crop.html
│   ├── fertilization.html
│   ├── result.html
│
├── static/
│   ├── style.css
│   └── images/
│       └── farm.jpg
│
├── dataset/
│   ├── crop_data.csv
│   ├── fertilizer_data.csv
│
└── README.md


Data Collection
      ↓
Data Cleaning & Preprocessing
      ↓
Label Encoding (Crop, Soil, Fertilizer)
      ↓
Feature Scaling (StandardScaler)
      ↓
Train-Test Split
      ↓
Model Training (RandomForestClassifier)
      ↓
Model Evaluation (Accuracy Score)
      ↓
Model Saving (.pkl files)
      ↓
Flask Integration
      ↓
Prediction Output (Web App)


Future Enhancements
🌦️ Live Weather API integration
📱 Mobile responsive UI
🗺️ Location-based farming suggestions
🤖 Deep Learning model upgrade
💾 Database integration (MySQL / MongoDB)
🌍 Multi-language support (Hindi + English)

Author

Name: Harsh goyal
Role: Machine Learning Developer
Project: Smart Farming AI System 🌾
