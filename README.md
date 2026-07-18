📊 Fitness App Analytics Assignment
A Data Analytics project that explores user behaviour in a fitness tracking mobile application using Exploratory Data Analysis (EDA), Machine Learning, and Data Visualization.
This project analyses user activity, demographics, app engagement, and health metrics to generate business insights that can help improve app performance and user engagement.
________________________________________
📌 Project Overview
A leading software development company wants to improve the performance of its fitness tracking mobile application. The dataset contains one year of user interaction and health-related information, including:
•	User demographics
•	Activity level
•	App usage
•	Distance travelled
•	Calories burned
•	User location
The objective is to analyse user behaviour and build predictive and clustering models that provide actionable insights.
________________________________________
🎯 Objectives
•	Perform Exploratory Data Analysis (EDA)
•	Understand user behaviour and engagement
•	Predict Calories Burned using Linear Regression
•	Predict Activity Level using Classification models
•	Segment users using K-Means Clustering
•	Evaluate machine learning models using appropriate metrics
•	Generate charts and reports for business decision-making
________________________________________
🗂 Dataset Features
Feature	Description
User ID	Unique identifier
Gender	Male / Female
Age	User age
Activity Level	Sedentary, Moderate, Active
Location	Urban, Rural, Suburban
App Sessions	Number of app sessions
Distance Travelled (km)	Distance travelled
Calories Burned	Calories burned
________________________________________
🛠 Technologies Used
•	Python 3
•	Pandas
•	NumPy
•	Matplotlib
•	Seaborn
•	Scikit-learn
•	JSON
________________________________________
📈 Project Workflow
1. Data Loading
•	Load dataset
•	Check dataset dimensions
•	Detect missing values
•	Detect duplicate records
________________________________________
2. Exploratory Data Analysis (EDA)
The project performs:
•	Descriptive statistics
•	Correlation analysis
•	Group-wise analysis by:
o	Activity Level
o	Gender
o	Location
Generated visualizations include:
•	Correlation Heatmap
•	Distribution Histograms
•	Scatter Plot
•	User Engagement Charts
________________________________________
3. Linear Regression
Goal
Predict Calories Burned
Features Used
•	App Sessions
•	Distance Travelled
•	Age
•	Gender
•	Location
Evaluation Metrics:
•	R² Score
•	RMSE
•	MAE
Visualization:
•	Actual vs Predicted Plot
________________________________________
4. Classification
Two machine learning models are compared.
•	Logistic Regression
•	Random Forest Classifier
Target Variable:
•	Activity Level
Evaluation Metrics:
•	Accuracy
•	Precision
•	Recall
•	F1 Score
Visualizations:
•	Confusion Matrix
•	Feature Importance
________________________________________
5. K-Means Clustering
User segmentation based on:
•	App Sessions
•	Distance Travelled
•	Calories Burned
The project includes:
•	Elbow Method
•	Silhouette Score
•	Cluster Visualization
•	Cluster Profile Analysis
________________________________________
## 📂 Project Structure

```text
fitness-app-analytics-assignment/
│
├── Classification/
│   ├── chart_confusion_matrix.png
│   └── chart_feature_importance.png
│
├── Exploratory_Data_Analysis_Charts/
│   ├── chart_correlation_heatmap.png
│   ├── chart_distributions.png
│   ├── chart_engagement_by_group.png
│   └── chart_sessions_vs_calories.png
│
├── K-Means/
│   ├── chart_cluster_profile.png
│   ├── chart_cluster_selection.png
│   └── chart_clusters.png
│
├── Linear_Regression_Chart/
│   └── chart_regression_actual_vs_pred.png
│
├── cluster_profile.csv
├── DA_Assessment2_analysis.py
├── dataset_raw.csv
├── dataset_with_clusters.csv
├── eda_correlation.csv
├── eda_describe.csv
├── eda_group_activity.csv
├── eda_group_gender.csv
├── eda_group_location.csv
├── results.json
├── README.md
├── requirements.txt
```________________________________________
📊 Outputs
The project generates:
CSV Files
•	eda_describe.csv
•	eda_correlation.csv
•	eda_group_activity.csv
•	eda_group_gender.csv
•	eda_group_location.csv
•	cluster_profile.csv
•	dataset_with_clusters.csv
Charts
•	Correlation Heatmap
•	Distribution Charts
•	Sessions vs Calories Scatter Plot
•	User Engagement Charts
•	Regression Prediction Plot
•	Confusion Matrix
•	Feature Importance
•	Elbow Method
•	Cluster Visualization
•	Cluster Profile
JSON Output
•	results.json
________________________________________
🚀 How to Run
Clone the repository
git clone https://github.com/saurabhgit2/fitness-app-analytics-assignment.git
Navigate into the project
cd fitness-app-analytics-assignment
Install dependencies
pip install -r requirements.txt
Run the analysis
python DA_Assessment2_analysis.py
________________________________________
📈 Machine Learning Models
Task	Algorithm
Regression	Linear Regression
Classification	Logistic Regression
Classification	Random Forest
Clustering	K-Means
________________________________________
📋 Evaluation Metrics
Regression
•	R² Score
•	RMSE
•	MAE
Classification
•	Accuracy
•	Precision
•	Recall
•	F1 Score
Clustering
•	Silhouette Score
•	Elbow Method
________________________________________
📚 Learning Outcomes
This project demonstrates practical experience in:
•	Data Cleaning
•	Exploratory Data Analysis
•	Statistical Analysis
•	Data Visualization
•	Feature Engineering
•	Regression
•	Classification
•	Clustering
•	Model Evaluation
•	Business Insight Generation
________________________________________
👨‍💻 Author
Saurabh Singh (Student ID-270732411)
Master of Software Engineering
Yoobee College of Creative Innovation
Auckland, New Zealand
________________________________________
📄 License
This project is developed for educational purposes as part of a Data Analytics assessment.

