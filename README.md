E-Commerce Data Analysis & Machine Learning Project

📌 Project Overview

This project focuses on analyzing E-Commerce transaction data to gain insights into customer behavior, purchasing patterns, and business strategies using Machine Learning. It covers customer segmentation, churn prediction, market basket analysis, revenue prediction, and customer lifetime value calculations.

📂 Dataset Information

Dataset Link: https://www.kaggle.com/datasets/carrie1/ecommerce-data

Number of Records: 522,698

Number of Features: 9

Key Features: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, TotalPrice

Data Cleaning & Preprocessing:

Removed duplicates

Handled missing values (Description, CustomerID imputed appropriately)

Excluded irrelevant transactions (e.g., bank charges, postage)

Created new feature: TotalPrice = Quantity × UnitPrice

Outlier handling using IQR-Based Capping, Winsorization, and Percentile-Based Capping

🔍 Problem Statements & Methodologies

1️⃣ Customer Churn Prediction

Goal: Identify customers who are likely to stop purchasing.

Methods:

Logistic Regression, Random Forest, XGBoost, SVM

Feature Engineering: Recency, Frequency, Monetary (RFM), purchase patterns

Evaluated using Accuracy, Precision, Recall, F1-score, AUC-ROC

Best Model: XGBoost (AUC-ROC: 0.91)

2️⃣ Customer Segmentation (Clustering)

Goal: Categorize customers into meaningful groups.

Methods:

K-Means, DBSCAN, Hierarchical Clustering

Features used: RFM metrics, purchasing frequency, basket size

Evaluated using Silhouette Score, Davies-Bouldin Index

Best Model: K-Means (Optimal K = 4, Silhouette Score = 0.73)

3️⃣ Predicting Customer Purchase Probability

Goal: Estimate the likelihood of a customer making a future purchase.

Methods:

Logistic Regression, Decision Trees, Neural Networks

Feature Engineering: Customer tenure, purchase patterns, seasonality trends

Evaluated using Log-Loss, ROC-AUC Score

Best Model: Decision Tree (ROC-AUC = 0.89)

4️⃣ Market Basket Analysis & Recommendation System

Goal: Identify frequently bought product combinations & suggest recommendations.

Methods:

Apriori Algorithm, FP-Growth for association rules

Created Recommendation System based on frequent itemsets

Results: Identified Top 10 frequently bought product pairs, optimized for cross-selling.

5️⃣ High-Value Customer Analysis

Goal: Identify top-spending customers and analyze their behavior.

Methods:

RFM Analysis (Recency, Frequency, Monetary Value)

Pareto Principle (80/20 rule for revenue distribution)

Results: Top 20% of customers contribute 75% of revenue.

6️⃣ Customer Lifetime Value (CLV) Prediction

Goal: Estimate future revenue from customers.

Methods:

Gamma-Gamma & Beta-Geometric models

Linear Regression, Decision Trees for CLV prediction

Best Model: Gamma-Gamma model (Predicted lifetime revenue within 5% error margin)

7️⃣ Total Price Prediction

Goal: Predict the total price of an order.

Methods:

Linear Regression with Polynomial Features (Degree = 2) (Final Model)

Compared with Ridge, Lasso, ElasticNet, XGBoost

Best Model: Polynomial Regression (R²: 0.98, RMSE: 0.1281)

8️⃣ Purchase Behavior Analysis

Goal: Understand purchasing trends across different customer segments.

Methods:

Time-Series Analysis (Seasonal trends, weekday/weekend purchases)

Customer segmentation & high-frequency item analysis

Results: Identified peak sales times, seasonality trends, and customer preferences.

📊 Results Summary

Problem Statement

Best Model/Method

Key Metric(s)

Customer Churn Prediction

XGBoost

AUC-ROC: 0.91

Customer Segmentation

K-Means (K=4)

Silhouette Score: 0.73

Purchase Probability

Decision Tree

ROC-AUC: 0.89

Market Basket Analysis

FP-Growth

Frequent Itemsets

High-Value Customers

RFM Analysis

80/20 Rule

CLV Prediction

Gamma-Gamma Model

5% error margin

Total Price Prediction

Polynomial Regression

R²: 0.98, RMSE: 0.1281

Purchase Behavior Analysis

Time-Series & Clustering

Seasonal Trends

🛠️ Tech Stack & Tools Used

Languages: Python

Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, TensorFlow

Machine Learning: Regression, Classification, Clustering, Association Rules

Data Visualization: Tableau, Matplotlib, Seaborn

Deployment: Streamlit (planned for future implementation)

📌 Future Improvements

🔹 Implement Deep Learning & NLP models for advanced text-based analysis.🔹 Deploy models using Flask/Streamlit for real-world application.🔹 Optimize computational efficiency for large-scale predictions.
