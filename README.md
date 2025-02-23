# HR Growth Lead Predictor-Forecasting High Potential Companies

## **Project Overview**
This project aims to predict **high-potential HR service leads** based on companies' **funding and hiring trends**. The dataset includes **synthetic data** to simulate real-world scenarios for developing and testing predictive models. The goal is to classify companies as **"hot leads"** using machine learning techniques.

---

## **Project Structure**

### **`main.ipynb`**
This Jupyter Notebook contains the entire **workflow of the project**, including:

- **Data Preprocessing**
  - Handling missing values
  - Extracting datetime-based features  
  - Dealing with infinite values  
  - Encoding categorical variables  
  - Standardizing numerical features  
  - Handling class imbalance using **SMOTE**  

- **Feature Engineering**
  - Creating new features from `last_funding_date`
  - Removing irrelevant columns  

- **Model Training**
  - Training multiple ML models including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost  

- **Model Evaluation**
  - Performance metrics (F1-score, Precision, Recall)
  - Confusion matrix analysis  

- **Hyperparameter Tuning**
  - Tuning XGBoost using **RandomizedSearchCV**  

### **Best Model: XGBoost**
After hyperparameter tuning, the best model is **XGBoost**, achieving the following performance:

| Metric      | Value   |
|------------|---------|
| **Accuracy**  | 0.972533 |
| **F1-Score**  | 0.902408 |
| **Precision** | 0.991827 |
| **Recall**    | 0.954424 |

This model was selected based on **F1-score, precision, and recall**, ensuring a balance between false positives and false negatives.

---

---

### **`saved_scaler/`**
This folder contains:
  - **`standard_scaler.pkl`** → The **trained StandardScaler object** used to standardize numerical features before training the model.  

---

### **`dataset/`**
This folder contains:
  - **`train.csv`** → Original training dataset  
  - **`test.csv`** → Original test dataset  
  - **`holdout.csv`** → Dataset for final prediction  
  - **`cleaned_train.csv`** → Preprocessed training data after cleaning, encoding, and scaling  

Each dataset has undergone **proper preprocessing steps**, ensuring **data consistency** across training, validation, and testing.

---

### **`best_model/`**
This folder contains:
  - **`best_xgb_model.pkl`** → The **best-performing XGBoost model** after hyperparameter tuning.  
  - This model was trained using optimized parameters to improve accuracy, F1-score, and recall.

---

### **`submission/`**
This folder contains:
  - **`submission.csv`** → The **final prediction file**, containing: company_id,is_hot_lead.
  - This file was generated using the best-trained model and is formatted according to the challenge submission requirements.

---

### **`project_report/`**
This file contains:
  - **Comprehensive project documentation** detailing:
  - **Problem statement**
  - **Data preprocessing steps**
  - **Feature engineering**
  - **Model training and evaluation**
  - **Hyperparameter tuning results**
  - **Final submission process**
  - **Refer to this document** to understand the complete project workflow.

