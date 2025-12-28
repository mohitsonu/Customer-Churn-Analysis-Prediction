# 📊 Customer Churn Analysis & Prediction

## 🎯 Objective
To analyze customer behavior and predict churn using machine learning techniques.

## 🛠️ Tools Used
- 🐍 **Python**
- 🐼 **Pandas**
- 🔢 **NumPy**
- 📊 **Matplotlib & Seaborn**
- 🤖 **Scikit-learn**

## 🔄 Workflow
1. 🧹 **Data Cleaning**
2. 🔍 **Exploratory Data Analysis (EDA)**
3. ⚙️ **Feature Engineering**
4. 🧠 **Model Training & Evaluation**

## 📈 Results
Logistic Regression and Random Forest models were trained and evaluated. Insights were generated to help reduce customer churn.

### 🏆 Model Performance
- **Logistic Regression Accuracy:** `81.55%`
- **Random Forest Accuracy:** `79.99%`

#### 📑 Classification Report (Logistic Regression)
```
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      1036
           1       0.69      0.56      0.62       373

    accuracy                           0.82      1409
   macro avg       0.77      0.73      0.75      1409
weighted avg       0.81      0.82      0.81      1409
```

---

## 📒 Notebook Outputs

### 1. 🧹 Data Cleaning (`1_data_cleaning.ipynb`)
**Dataset Info:**
```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 20 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   gender            7043 non-null   object 
 1   SeniorCitizen     7043 non-null   int64  
 2   Partner           7043 non-null   object 
 3   Dependents        7043 non-null   object 
 ...
 18  TotalCharges      7043 non-null   object 
 19  Churn             7043 non-null   int64  
dtypes: float64(1), int64(3), object(16)
memory usage: 1.1+ MB
```

### 2. 📊 EDA (`2_eda.ipynb`)
*(Visualizations included in notebook)*

### 3. ⚙️ Feature Engineering (`3_feature_engineering.ipynb`)
**Processed Data Preview:**
```text
   gender  SeniorCitizen  Partner  Dependents    tenure  PhoneService  ...
0       0              0        1           0 -1.277445             0  ...
1       1              0        0           0  0.066327             1  ...
2       1              0        0           0 -1.236724             1  ...
3       1              0        0           0  0.514251             0  ...
4       0              0        0           0 -1.236724             1  ...
```

### 4. 🧠 Model Training (`4_model_training.ipynb`)
**Training Results:**
- **Logistic Regression Accuracy:** `81.69%`
- **Random Forest Accuracy:** `79.84%`

**Classification Report (LR):**
```
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      1036
           1       0.69      0.57      0.62       373

    accuracy                           0.82      1409
   macro avg       0.77      0.74      0.75      1409
weighted avg       0.81      0.82      0.81      1409
```
