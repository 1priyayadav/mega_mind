# Coronary Heart Disease Prediction using Hybrid 1D-CNN-LightGBM

This repository contains the code and documentation for a novel hybrid machine learning approach to predicting Coronary Heart Disease (CHD). 

Based on the 2025 Springer *Journal of Big Data* publication *"An early and accurate diagnosis and detection of the coronary heart disease using deep learning and machine learning algorithms"*, this project successfully improves upon standalone models (SVM, XGBoost) by introducing a 1-Dimensional Convolutional Neural Network (1D-CNN) as an automated feature extractor feeding into a LightGBM classifier.

## Repository Contents
* **`download_data.py`**: Python script to fetch the UCI Heart Disease dataset using the `ucimlrepo` library and format it into a CSV.
* **`run_experiments.py`**: The core execution script. It preprocesses the data using SMOTE and StandardScaling, trains the baseline and proposed hybrid models, and evaluates their performance against critical medical metrics.
* **`output/`**: Directory containing generated visualization matrices from the experiments (ROC curves, confusion matrices, performance bar charts).

## How to Run the Code
1. Clone the repository and navigate to the project directory.
2. Ensure you have Python 3.x installed.
3. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn torch ucimlrepo
   ```
4. Download the dataset:
   ```bash
   python download_data.py
   ```
5. Run the models and generate metrics:
   ```bash
   python run_experiments.py
   ```
   *The script will output the accuracy, precision, recall, and F1-scores to your console while saving charts to an automatically generated `output/` folder.*

## Results Summary
The hybrid **1D-CNN-LightGBM** model achieved a **Recall of 83.33%** and an **F1-Score of 78.95%**, significantly outperforming the standalone SVM and XGBoost baselines in correctly identifying true-positive disease classifications.
