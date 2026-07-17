"""
DA Assessment 2 - Task 1: Advanced Statistical Analysis for Software Engineering
Fitness Tracking App Dataset Analysis

Author: Saurabh Singh (Student ID: 270732411)

This script performs:
  1. Exploratory Data Analysis (EDA)
  2. Linear Regression - predicting Calories Burned (continuous target)
  3. Classification (Logistic Regression + Random Forest) - predicting Activity Level
  4. K-Means Clustering - user segmentation for engagement analysis
  5. Model evaluation with appropriate metrics (R2, RMSE, MAE, Accuracy,
     Precision, Recall, F1, Silhouette Score)
  6. Visualisations saved as PNG files for inclusion in the written report
  """

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="viridis")
RANDOM_STATE = 42

results = {}  # collected metrics, written to results.json at the end

# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("dataset_raw.csv")

results["n_rows"] = int(df.shape[0])
results["n_cols"] = int(df.shape[1])
results["missing_values"] = int(df.isnull().sum().sum())
results["duplicate_rows"] = int(df.duplicated().sum())
