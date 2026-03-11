import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create an output directory for plots
import os
os.makedirs('output', exist_ok=True)

# 1. Load Data
df = pd.read_csv('heart_disease.csv')
X = df.drop(columns=['target'])
y = df['target']

# 2. Preprocessing
print("Preprocessing dataset...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Dictionary to store results
results = {}

def evaluate_model(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'AUC': auc}
    print(f"\n[{name}] Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'output/cm_{name}.png')
    plt.close()
    
    # return None
    
roc_data = {}

# 3. Base Model 1: Support Vector Machine (SVM)
print("\nTraining SVM...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
start_time = time.time()
svm_model.fit(X_train, y_train)
svm_time = time.time() - start_time
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_svm)
roc_auc = roc_auc_score(y_test, y_prob_svm)
roc_data['SVM'] = (fpr, tpr, roc_auc)
evaluate_model('SVM', y_test, y_pred_svm, y_prob_svm)

# 4. Base Model 2: XGBoost
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
start_time = time.time()
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
roc_auc = roc_auc_score(y_test, y_prob_xgb)
roc_data['XGBoost'] = (fpr, tpr, roc_auc)
evaluate_model('XGBoost', y_test, y_pred_xgb, y_prob_xgb)

# 5. Proposed Model: Hybrid 1D-CNN + LightGBM
print("\nTraining Proposed Model: 1D-CNN feature extractor + LightGBM...")

# Prepare data for PyTorch CNN
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) # shape: (batch, channels, length)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Calculate size after pooling
        self.flatten_size = 32 * (input_size // 2 // 2) 
        if self.flatten_size == 0:
            self.flatten_size = 32 * (input_size) # Fallback for very small feature sets

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.flatten_size, 2)
        
        self.feature_extractor = nn.Sequential(
            self.conv1, self.relu, self.maxpool,
            self.conv2, self.relu, nn.MaxPool1d(kernel_size=2) if input_size >= 4 else nn.Identity(),
            self.flatten
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc(features)
        return output, features

input_features = X_train.shape[1]
cnn_model = CNN1D(input_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Pre-train CNN
epochs = 20
cnn_model.train()
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs, _ = cnn_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Extract CNN features
cnn_model.eval()
with torch.no_grad():
    _, X_train_cnn = cnn_model(X_train_tensor)
    _, X_test_cnn = cnn_model(X_test_tensor)

X_train_cnn_np = X_train_cnn.numpy()
X_test_cnn_np = X_test_cnn.numpy()

# Train LightGBM on CNN features
lgb_model = lgb.LGBMClassifier(random_state=42)
start_time = time.time()
lgb_model.fit(X_train_cnn_np, y_train)
lgb_time = time.time() - start_time
y_pred_lgbm = lgb_model.predict(X_test_cnn_np)
y_prob_lgbm = lgb_model.predict_proba(X_test_cnn_np)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob_lgbm)
roc_auc = roc_auc_score(y_test, y_prob_lgbm)
roc_data['1D-CNN-LightGBM'] = (fpr, tpr, roc_auc)
evaluate_model('1D-CNN-LightGBM', y_test, y_pred_lgbm, y_prob_lgbm)

# 6. Performance Visualizations
# Plot ROC Curves together
plt.figure(figsize=(8, 6))
for label, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.legend(loc="lower right")
plt.savefig('output/roc_comparison.png')
plt.close()

# Plot Performance Metrics Comparison Bar Chart
metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim([0.7, 1.0])
plt.xticks(rotation=45)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('output/metrics_comparison.png')
plt.close()

# Save metrics to CSV for the report
metrics_df.to_csv('output/model_performance_metrics.csv')

print("\nExperiments complete! Plots and metrics saved to 'output/' folder.")
