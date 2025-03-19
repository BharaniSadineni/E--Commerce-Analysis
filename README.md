**E-Commerce Data Analysis & Machine Learning Project**

📌 **Project Overview**

This project focuses on analyzing E-Commerce transaction data to gain insights into customer behavior, purchasing patterns, and business strategies using Machine Learning. It covers customer segmentation, churn prediction, market basket analysis, revenue prediction, and customer lifetime value calculations.

📂 **Dataset Information**

Dataset Link: https://www.kaggle.com/datasets/carrie1/ecommerce-data

Number of Records: 541,909

Number of Features: 8

Key Features: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

Data Cleaning & Preprocessing:

Removed duplicates

Handled missing values (Description(Mode), CustomerID imputed appropriately(Inetrpolation, forward and backwardfill))

Excluded irrelevant transactions (e.g., bank charges, postage)

Created new feature: TotalPrice = Quantity × UnitPrice

Outlier handling using IQR-Based Capping, Winsorization, and Percentile-Based Capping

🔍 **Problem Statements & Methodologies**

1️⃣** Customer Churn Prediction**

Goal: Identify customers who are likely to stop purchasing.

Methods:

Logistic Regression, Random Forest, XGBoost, CATBoost

Feature Engineering: Recency, Frequency, Monetary (RFM), purchase patterns

Evaluated using Accuracy, Precision, Recall, F1-score, AUC-ROC

Best Model: XGBoost (AUC-ROC: 0.91)

2️⃣ **Customer Segmentation (Clustering)**

Goal: Categorize customers into meaningful groups.

Methods:

K-Means

Features used: RFM metrics, purchasing frequency

Evaluated using Silhouette Score, Davies-Bouldin Index

Best Model: K-Means (Optimal K = 4, Silhouette Score = 0.76)

3️⃣ **Predicting Customer Purchase Probability**

Goal: Estimate the likelihood of a customer making a future purchase.

Methods:

Logistic Regression, Random Forest, XGBoost, CATBoost

Feature Engineering: Customer tenure, purchase patterns, seasonality trends

Balcancing heavy class imbalance by assing calss weights and Evaluated using Log-Loss, ROC-AUC Score

Best Model: CatBoost (ROC-AUC = 0.9658)

4️⃣ **Market Basket Analysis & Recommendation System**

Goal: Identify frequently bought product combinations & suggest recommendations.

Methods:

Apriori Algorithm for association rules

Created Recommendation System based on frequent itemsets

Logistic Regression, Random Forest, XGBoost, CATBoost

 Assess model performance using accuracy, precision, recall, and ROC-AUC.

5️⃣ **High-Value Customer Analysis**

Goal: Identify top-spending customers and analyze their behavior.

Methods:

RFM Analysis (Recency, Frequency, Monetary Value)

Creating High value purhcase binary columns

Best Model: Random Forest (Accuracy: 97.77%, AUC-ROC: 0.9966)

6️⃣ **Customer Lifetime Value (CLV) Prediction**

Goal: Estimate future revenue from customers.

Methods:

Feature Engineering:

Average Purchase Value: Calculated by dividing the total monetary value by the frequency of purchases.

Customer Lifespan: Assumed to be 365 days for the analysis.

Customer Lifetime Value (CLV): Derived using the formula: [ \text{CLV} = \text{Average Purchase Value} \times \text{Frequency} \times \text{Customer Lifespan} ]

Models: Random Forest, XGBoost, CATBoost regressors

evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2)

Best Model: CatBoost (R2:99.05)

7️⃣ **Total Price Prediction**

Goal: Predict the total price of an order.

Methods:

Linear Regression with Polynomial Features (Degree = 2) (Final Model)

Compared with Ridge, Lasso, ElasticNet, XGBoost

Best Model: Polynomial Regression (R²: 0.98, RMSE: 0.1281)

📊** Results Summary**

Problem Statement

Best Model/Method

Key Metric(s)

**Customer Churn Prediction**

XGBoost

AUC-ROC: 0.91

**Customer Segmentation**

K-Means (K=4)

Silhouette Score: 0.76

**Purchase Probability**

CatBoost

ROC-AUC: 0.96

**Market Basket Analysis**

Apriori

Frequent Itemsets

**High-Value Customers**

Random Forest 

Accuracy: 97.77%, AUC-ROC: 0.9966

**CLV Prediction**

Avergae Purchase Value, Customer Life Span, CLV

CatBoost 

R2:99.05

**Total Price Prediction**

Polynomial Regression

R²: 0.98, RMSE: 0.1281


🛠️ **Tech Stack & Tools Used**

Languages: Python

Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, CatBoost, Optuna, Matplotlib, Seaborn, TensorFlow

Machine Learning: Regression, Classification, Clustering, Association Rules

Data Visualization: Matplotlib, Seaborn

Deployment: Streamlit (planned for future implementation)

📌 Future Improvements

🔹 Implement Deep Learning & NLP models for advanced text-based analysis.🔹 Deploy models using Flask/Streamlit for real-world application.🔹 Optimize computational efficiency for large-scale predictions.
