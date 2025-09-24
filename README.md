# Breast Cancer Analysis & AI/ML Prediction

## Project Overview
This project performs **statistical analysis** and **machine learning prediction** on the **Breast Cancer Wisconsin (Diagnostic) dataset**. The goal is to analyze tumor features, visualize trends, perform hypothesis testing, and build a model to classify tumors as **benign** or **malignant**.

---

## Features
- **Data Cleaning & Preprocessing**: Handles missing values and standardizes column names.
- **Descriptive Statistics**: Summary of features including mean, median, and standard deviation.
- **Data Visualization**:
  - Count of benign vs malignant tumors
  - Feature correlation heatmap
  - Feature importance chart
- **Hypothesis Testing**: Compares mean radius between benign and malignant tumors using t-test.
- **Machine Learning Model**: Random Forest Classifier to predict tumor diagnosis.
- **Evaluation Metrics**:
  - Accuracy
  - Classification report
  - Confusion matrix
- **PDF Report Generation**: Automatically generates a formatted report including plots and results.

---

## Files in the Project
| File | Description |
|------|-------------|
| `breast_cancer_data.csv` | Dataset containing tumor features and diagnosis labels. |
| `breast_cancer_project.py` | Python script for analysis, ML modeling, and PDF report generation. |
| `plots/` | Folder containing all saved visualizations (count plot, correlation heatmap, feature importance). |
| `Breast_Cancer_Analysis_Report.pdf` | Generated PDF report summarizing analysis, results, and visualizations. |

---

## Prerequisites
Install the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy fpdf

**## How to Run**
Place breast_cancer_data.csv and breast_cancer_project.py in the same folder.

Open terminal and navigate to the project folder:
cd "PROJECT2"

Run the Python script:
python breast_cancer_project.py


**## Outputs:**
Plots will be saved in the plots/ folder.
PDF report Breast_Cancer_Analysis_Report.pdf will be generated in the project folder.
Terminal will display dataset columns, first 5 rows, and other intermediate outputs.

**Results:**
High model accuracy in predicting benign vs malignant tumors.
Key features influencing prediction: radius_mean, texture_mean, concavity_mean.
Statistical tests show significant differences between benign and malignant tumors.

Author:
Axilia Jennifer B  – B.Tech Student, Christ (Deemed to be University)

License
This project is for academic purposes only.


---
