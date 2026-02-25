import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import joblib

d = pd.read_csv("/Users/shivanshsahu/Downloads/data.csv")
print(d.sample(5))
print('\n')
print(d.head())
print('\n')
print(d.shape)
print("Columns in my CSV:", d.columns.tolist())

if 'Unnamed: 32' in d.columns:
    d.drop('Unnamed: 32', axis=1, inplace=True)

d = d.drop(columns = ['perimeter_mean' , 'area_mean' , 'perimeter_se' , 'area_se' , 'perimeter_worst' , 'area_worst'])

print(d.shape)
    
print(d.describe())
print(d.describe().transpose())
print(d.isnull().sum())


d.rename(columns={'diagnosis' : 'target'}, inplace=True)

print(d.corr(numeric_only=True))

d['target'] = d['target'].map({'M': 1, 'B': 0})

import os

if not os.path.exists('images'):
    os.makedirs('images')

print("Generating Target Distribution Plot...")

plt.figure(figsize=(8, 5))

ax = sns.countplot(x = 'target', data=d, palette='Set2')

plt.title('Distribution of Cell Diagnoses', fontsize=14)
plt.xlabel('Diagnosis (0: Malignant, 1: Benign)', fontsize=12)
plt.ylabel('Number of Cell Samples', fontsize=12)

for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.4, p.get_height()), 
                ha='center', va='top', color='white', size=12)

plt.savefig('images/01_class_distribution.png', dpi=300, bbox_inches='tight')
print("Plot saved to images/01_class_distribution.png")

plt.show()

print("\nGenerating Correlation Heatmap...")

if 'id' in d.columns:
    d.drop('id', axis=1, inplace=True)

corr_matrix = d.corr(numeric_only=True)

plt.figure(figsize=(10, 6))

sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

plt.title('Biological Feature Correlation Heatmap', fontsize=18)

plt.savefig('images/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Plot saved to images/02_correlation_heatmap.png")

plt.show()

X = d.drop(columns = ['target'])
Y = d['target']

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2, random_state=42)
print(X_train.shape , Y_train.shape , X_test.shape , Y_test.shape)

#making pipeline'
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lor', LogisticRegression(max_iter=2500))
])

pipe.fit(X_train , Y_train)

Y_pred = pipe.predict(X_test)

print(classification_report(Y_test , Y_pred))
#print("r2_score = " ,  r2_score(Y_test , Y_pred))

cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Benign', 'Predicted Malignant'], 
            yticklabels=['Actual Benign', 'Actual Malignant'])

plt.title('Clinical Confusion Matrix')
plt.ylabel('True Diagnosis')
plt.xlabel('Algorithm Prediction')

plt.savefig('images/03_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion Matrix saved to images/03_confusion_matrix.png")

plt.show()

from sklearn.metrics import roc_curve, auc

print("\n--- Phase 6: Clinical Interpretation (ROC Curve) ---")

y_prob = pipe.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, y_prob)

roc_auc = auc(fpr, tpr)
print(f"ROC-AUC Score: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Healthy cells flagged incorrectly)')
plt.ylabel('True Positive Rate (Tumors caught successfully)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.savefig('images/04_roc_curve.png', dpi=300, bbox_inches='tight')
print("ROC Curve saved to images/04_roc_curve.png")
plt.show()

joblib.dump(pipe, 'model.pkl')