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


# ---------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
numeric_cols = ["Age", "App Sessions", "Distance Travelled (km)", "Calories Burned"]

desc = df[numeric_cols].describe().round(2)
desc.to_csv("eda_describe.csv")

corr = df[numeric_cols].corr().round(3)
corr.to_csv("eda_correlation.csv")
results["correlation_matrix"] = corr.to_dict()

group_activity = df.groupby("Activity Level")[numeric_cols].mean().round(2)
group_activity.to_csv("eda_group_activity.csv")

group_gender = df.groupby("Gender")[numeric_cols].mean().round(2)
group_gender.to_csv("eda_group_gender.csv")

group_location = df.groupby("Location")[numeric_cols].mean().round(2)
group_location.to_csv("eda_group_location.csv")

# --- Chart 1: correlation heatmap ---
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="viridis", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("Exploratory_Data_Analysis_Charts/chart_correlation_heatmap.png", dpi=150)
plt.close()

# --- Chart 2: distribution of key numeric variables ---
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#3b6fa0")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("Exploratory_Data_Analysis_Charts/chart_distributions.png", dpi=150)
plt.close()

# --- Chart 3: App Sessions vs Calories Burned by Activity Level ---
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df, x="App Sessions", y="Calories Burned",
    hue="Activity Level", alpha=0.5, s=20
)
plt.title("App Sessions vs Calories Burned by Activity Level")
plt.tight_layout()
plt.savefig("Exploratory_Data_Analysis_Charts/chart_sessions_vs_calories.png", dpi=150)
plt.close()

# --- Chart 4: engagement by demographic group ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
group_activity["App Sessions"].plot(kind="bar", ax=axes[0], color="#3b6fa0")
axes[0].set_title("Avg. App Sessions by Activity Level")
axes[0].set_ylabel("App Sessions")
group_gender["App Sessions"].plot(kind="bar", ax=axes[1], color="#5a9367")
axes[1].set_title("Avg. App Sessions by Gender")
group_location["App Sessions"].plot(kind="bar", ax=axes[2], color="#b0703b")
axes[2].set_title("Avg. App Sessions by Location")
for ax in axes:
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("Exploratory_Data_Analysis_Charts/chart_engagement_by_group.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. LINEAR REGRESSION - Predicting Calories Burned
#    (predictors: App Sessions, Distance Travelled, Age, Gender, Location)
# ---------------------------------------------------------------------------
reg_df = df.copy()
reg_df = pd.get_dummies(reg_df, columns=["Gender", "Location"], drop_first=True)

feature_cols_reg = [
    "App Sessions", "Distance Travelled (km)", "Age",
    "Gender_Male", "Location_Suburban", "Location_Urban",
]
X_reg = reg_df[feature_cols_reg]
y_reg = reg_df["Calories Burned"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)

reg_metrics = {
    "r2_score": round(r2_score(y_test, y_pred), 4),
    "rmse": round(mean_squared_error(y_test, y_pred) ** 0.5, 2),
    "mae": round(mean_absolute_error(y_test, y_pred), 2),
    "coefficients": dict(zip(feature_cols_reg, np.round(lin_model.coef_, 3))),
    "intercept": round(float(lin_model.intercept_), 3),
}
results["regression"] = reg_metrics

# --- Chart 5: actual vs predicted ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4, s=18, color="#3b6fa0")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", lw=1.5)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title(f"Linear Regression: Actual vs Predicted (R2 = {reg_metrics['r2_score']})")
plt.tight_layout()
plt.savefig("Linear_Regression_Chart/chart_regression_actual_vs_pred.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. CLASSIFICATION - Predicting Activity Level
#    (Logistic Regression + Random Forest, compared)
# ---------------------------------------------------------------------------
clf_df = df.copy()
le = LabelEncoder()
clf_df["Activity_Level_Enc"] = le.fit_transform(clf_df["Activity Level"])
clf_df = pd.get_dummies(clf_df, columns=["Gender", "Location"], drop_first=True)

feature_cols_clf = [
    "App Sessions", "Distance Travelled (km)", "Calories Burned", "Age",
    "Gender_Male", "Location_Suburban", "Location_Urban",
]
X_clf = clf_df[feature_cols_clf]
y_clf = clf_df["Activity_Level_Enc"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

scaler = StandardScaler()
Xc_train_s = scaler.fit_transform(Xc_train)
Xc_test_s = scaler.transform(Xc_test)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(Xc_train_s, yc_train)
yc_pred_lr = logreg.predict(Xc_test_s)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(Xc_train, yc_train)
yc_pred_rf = rf.predict(Xc_test)

def clf_metrics(y_true, y_pred, label):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return {
        "model": label,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
    }

results["classification"] = {
    "logistic_regression": clf_metrics(yc_test, yc_pred_lr, "Logistic Regression"),
    "random_forest": clf_metrics(yc_test, yc_pred_rf, "Random Forest"),
    "classes": list(le.classes_),
}

# --- Chart 6: confusion matrix for the better model (Random Forest) ---
cm = confusion_matrix(yc_test, yc_pred_rf)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Activity Level")
plt.ylabel("Actual Activity Level")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("Classification/chart_confusion_matrix.png", dpi=150)
plt.close()

# --- Chart 7: feature importance (Random Forest) ---
importances = pd.Series(rf.feature_importances_, index=feature_cols_clf).sort_values()
plt.figure(figsize=(7, 4.5))
importances.plot(kind="barh", color="#3b6fa0")
plt.title("Random Forest Feature Importance (Activity Level Prediction)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("Classification/chart_feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. K-MEANS CLUSTERING - User Segmentation
# ---------------------------------------------------------------------------
cluster_features = ["App Sessions", "Distance Travelled (km)", "Calories Burned"]
X_cluster = df[cluster_features]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Elbow method
inertias = []
sil_scores = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_k = km.fit_predict(X_cluster_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_scaled, labels_k))

plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.plot(list(K_range), inertias, marker="o", color="#3b6fa0")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.subplot(1, 2, 2)
plt.plot(list(K_range), sil_scores, marker="o", color="#b0703b")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by k")
plt.tight_layout()
plt.savefig("K-Means/chart_cluster_selection.png", dpi=150)
plt.close()

optimal_k = 3  # chosen from elbow + silhouette + interpretability
kmeans_final = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
df["Cluster"] = kmeans_final.fit_predict(X_cluster_scaled)
final_sil = silhouette_score(X_cluster_scaled, df["Cluster"])

cluster_profile = df.groupby("Cluster")[cluster_features + ["Age"]].mean().round(1)
cluster_profile["Count"] = df["Cluster"].value_counts().sort_index()
cluster_profile.to_csv("cluster_profile.csv")

results["clustering"] = {
    "optimal_k": optimal_k,
    "silhouette_score": round(final_sil, 4),
    "cluster_profile": cluster_profile.to_dict(orient="index"),
}

# --- Chart 8: cluster visualisation ---
plt.figure(figsize=(7, 5.5))
sns.scatterplot(
    data=df, x="App Sessions", y="Calories Burned",
    hue="Cluster", palette="viridis", alpha=0.6, s=22
)
plt.title(f"User Segments from K-Means Clustering (k={optimal_k}, silhouette={final_sil:.3f})")
plt.tight_layout()
plt.savefig("K-Means/chart_clusters.png", dpi=150)
plt.close()

# --- Chart 9: cluster profile bar chart ---
plt.figure(figsize=(8, 5))
cluster_profile[cluster_features].plot(kind="bar", ax=plt.gca())
plt.title("Average Metrics per User Segment")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("K-Means/chart_cluster_profile.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# SAVE ALL RESULTS
# ---------------------------------------------------------------------------
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

df.to_csv("dataset_with_clusters.csv", index=False)

print("Analysis complete. Results written to results.json")
print(json.dumps(results, indent=2, default=str))
