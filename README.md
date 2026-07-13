# 🩺 Diabetes Prediction System

A Machine Learning project that predicts whether a patient is likely to have diabetes using health-related medical measurements from the **Pima Indians Diabetes Dataset**. The project covers the complete ML pipeline, from data preprocessing and model training to deployment using **Streamlit**.

---

## 📌 Project Overview

This project demonstrates the end-to-end process of building a machine learning model for diabetes prediction.

The workflow includes:

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Training multiple machine learning models
* Hyperparameter tuning using GridSearchCV
* Model evaluation
* Feature importance analysis
* Deployment with Streamlit

---

## 🚀 Features

* Predicts diabetes based on 8 medical parameters
* Compares multiple machine learning algorithms
* Hyperparameter tuning for improved performance
* Interactive Streamlit web application
* Clean and beginner-friendly implementation

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

---

## 📂 Project Structure

```text
Diabetes_Prediction/
│
├── Predicting_Diabetes.ipynb      # Model training and analysis
├── app.py                         # Streamlit application
├── diabetes_model.joblib          # Trained model
├── scaler.joblib                  # Saved scaler
├── diabetes.csv                   # Dataset
├── README.md
```

---

## 📊 Dataset

This project uses the **Pima Indians Diabetes Dataset**.

### Features

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

### Target

* **Outcome**

  * 1 → Diabetic
  * 0 → Non-Diabetic

---

## ⚙️ Machine Learning Workflow

### 1. Data Cleaning

Some medical attributes contained invalid values (0), representing missing data. These values were replaced using appropriate preprocessing techniques.

### 2. Feature Engineering

* Log transformation applied to skewed features
* Feature scaling using **MinMaxScaler**

### 3. Model Training

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gaussian Naive Bayes

### 4. Hyperparameter Tuning

The best-performing model (**Random Forest**) was optimized using **GridSearchCV**.

### 5. Feature Importance

Feature importance was analyzed to identify the most influential medical attributes affecting diabetes prediction.

### 6. Deployment

The trained model and scaler were saved using **Joblib** and deployed through a **Streamlit** web application.

---

## 📈 Model Performance

The Random Forest model achieved the best overall performance after hyperparameter tuning and was selected for deployment.

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Diabetes_Prediction.git
```

Move into the project directory:

```bash
cd Diabetes_Prediction
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

Run the application:

```bash
streamlit run app.py
```

---

## 📷 Application

The Streamlit application allows users to:

* Enter patient medical details
* Predict diabetes in real time
* Display the prediction instantly

---

## 📌 Disclaimer

This project is developed for educational and portfolio purposes only. It is **not** intended for medical diagnosis or clinical decision-making.

---

## 👩‍💻 Author

**Tamanna Parmar**


GitHub: https://github.com/Tamannaahh


