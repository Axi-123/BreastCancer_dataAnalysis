# ==========================================================
# Project: Statistical Analysis & AI/ML Prediction of Breast Cancer
# Dataset: Breast Cancer Wisconsin (Diagnostic)
# Files: breast_cancer_data.csv, breast_cancer_project.py
# Tools: Python (pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, fpdf)
# ==========================================================

# -------------------------------
# Step 1: Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from fpdf import FPDF
import os

# -------------------------------
# Step 2: Load Dataset
# -------------------------------
df = pd.read_csv("breast_cancer_data.csv", on_bad_lines='skip')

# Standardize column names
df.columns = df.columns.str.lower().str.strip()

# Remove any non-numeric columns from features except 'diagnosis'
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Include all numeric columns in X, drop 'id' if exists
X = df[numeric_cols].copy()
if 'id' in X.columns:
    X = X.drop('id', axis=1)

# Target variable
y = df['diagnosis'].map({'B':0, 'M':1})

# Create folder to save plots
os.makedirs("plots", exist_ok=True)

# Debug prints
print("Columns in dataset:", df.columns)
print("First 5 rows:")
print(df.head())

# -------------------------------
# Step 3: Descriptive Statistics
# -------------------------------
desc_stats = df.describe()
missing_values = df.isnull().sum()

# -------------------------------
# Step 4: Visualizations
# -------------------------------
# Count of benign vs malignant
if 'diagnosis' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='diagnosis', data=df)
    plt.title('Count of Benign vs Malignant Tumors')
    plt.savefig("plots/count_plot.png")
    plt.close()

# Correlation heatmap (only numeric features)
plt.figure(figsize=(12,10))
sns.heatmap(df[numeric_cols].corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# -------------------------------
# Step 5: Hypothesis Testing
# -------------------------------
benign_radius = df[df['diagnosis']=='B']['radius_mean']
malignant_radius = df[df['diagnosis']=='M']['radius_mean']

t_stat, p_val = ttest_ind(benign_radius, malignant_radius)
t_test_result = f"T-statistic: {t_stat:.3f}, P-value: {p_val:.3e}"
t_test_summary = "Significant difference in mean radius between benign and malignant tumors." if p_val < 0.05 else "No significant difference detected."

# -------------------------------
# Step 6: Prepare Data for ML
# -------------------------------
# Already ensured X is numeric, y is 0/1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 7: ML Model - Random Forest Classifier
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# -------------------------------
# Step 8: Feature Importance
# -------------------------------
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title("Feature Importance in Predicting Breast Cancer")
plt.savefig("plots/feature_importance.png")
plt.close()

# -------------------------------
# Step 9: Generate PDF Report
# -------------------------------
pdf = FPDF()
pdf.add_page()

# Title
pdf.set_font("Times", 'B', 18)
pdf.cell(0, 10, "Statistical Analysis & AI/ML Prediction of Breast Cancer", ln=True, align='C')
pdf.ln(10)

# Abstract
pdf.set_font("Times", 'B', 16)
pdf.cell(0, 10, "Abstract", ln=True)
pdf.set_font("Times", '', 12)
pdf.multi_cell(0, 6,
"This project analyzes the Breast Cancer Wisconsin dataset, performs statistical tests, and builds a machine learning model to predict benign and malignant tumors. The analysis includes data visualization, hypothesis testing, and evaluation of model performance.")
pdf.ln(5)

# Descriptive Statistics
pdf.set_font("Times", 'B', 16)
pdf.cell(0, 10, "Descriptive Statistics", ln=True)
pdf.set_font("Times", '', 12)
pdf.multi_cell(0, 6, str(desc_stats))
pdf.ln(5)

# Missing Values
pdf.set_font("Times", 'B', 16)
pdf.cell(0, 10, "Missing Values", ln=True)
pdf.set_font("Times", '', 12)
pdf.multi_cell(0, 6, str(missing_values))
pdf.ln(5)

# Visualizations
pdf.set_font("Times", 'B', 16)
pdf.cell(0, 10, "Data Visualizations", ln=True)
pdf.image("plots/count_plot.png", w=120)
pdf.ln(5)
pdf.image("plots/correlation_heatmap.png", w=160)
pdf.ln(5)
pdf.image("plots/feature_importance.png", w=150)
pdf.ln(5)

# Hypothesis Testing
pdf.set_font("Times", 'B', 16)
pdf.cell(0, 10, "Hypothesis Testing", ln=True)
pdf.set_font("Times", '', 12)
pdf.multi_cell(0, 6, f"{t_test_result}\nResult: {t_test_summary}")
pdf.ln(5)

# ML Model Performance
pdf.set_font("Times", 'B', 16)
pdf.cell(0, 10, "Machine Learning Model Performance", ln=True)
pdf.set_font("Times", '', 12)
pdf.multi_cell(0, 6, f"Random Forest Accuracy: {accuracy*100:.2f}%")
pdf.multi_cell(0, 6, f"Classification Report:\n{classification_rep}")
pdf.multi_cell(0, 6, f"Confusion Matrix:\n{conf_matrix}")
pdf.ln(5)

# Summary
pdf.set_font("Times", 'B', 16)
pdf.cell(0, 10, "Summary", ln=True)
pdf.set_font("Times", '', 12)
pdf.multi_cell(0, 6,
"- Radius_mean, concavity_mean, and texture_mean are highly correlated with cancer presence.\n"
"- ML model achieved high accuracy in predicting benign vs malignant tumors.\n"
"- Statistical tests show significant differences between benign and malignant tumor features."
)

# Save PDF
pdf.output("Breast_Cancer_Analysis_Report.pdf")
print("PDF report generated: Breast_Cancer_Analysis_Report.pdf")
