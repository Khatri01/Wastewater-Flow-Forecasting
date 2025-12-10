# Wastewater Flow Forecasting

## Overview
This repository contains the end-to-end workflow for a IOT-Cloud-machine learning-based wastewater flow prediction system. It includes resources for both **local model training** and **production deployment** using a microservices architecture. 

## Repository Contents

### 1. Local Training & Analysis (`ML Models/`)
* Contains the Python scripts and Jupyter notebooks used to train, test, and validate the models locally.
* Includes data preprocessing, hyperparameter tuning, and performance evaluation for all five algorithms.

### 2. Prediction Deployment Services
Standalone deployment code for five distinct models, containerized as individual microservices:
* **Deep Learning:** `cnn-prediction-service`, `gru-prediction-service`, and `lstm-prediction-service`
* **Ensemble ML:** `rf-prediction-service` (Random Forest) and `xgboost-prediction-service`

### 3. Visualization (`Dashboard-service/`)
* Source code for the web-based dashboard that aggregates predictions and visualizes flow data for real-time monitoring.

