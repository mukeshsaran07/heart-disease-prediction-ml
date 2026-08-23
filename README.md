# ❤️ Heart Disease Prediction using Machine Learning

A machine learning classification project that predicts the possibility of heart disease based on patient medical information.

The project covers the complete machine learning workflow, including **data exploration, preprocessing, feature encoding, feature scaling, model comparison, model selection, and deployment using Streamlit**.

---

## 📌 Project Overview

Heart disease is one of the major health-related problems worldwide. The objective of this project is to build a machine learning model that can classify whether a patient is likely to have heart disease based on medical and clinical features.

The project compares multiple machine learning algorithms and selects **Logistic Regression** as the final model based on its performance.

A **Streamlit web application** is also developed to allow users to enter patient information and receive a prediction.

---

## 📊 Dataset

The dataset contains **918 records and 12 columns**.

### Features

| Feature        | Description                           |
| -------------- | ------------------------------------- |
| Age            | Age of the patient                    |
| Sex            | Gender of the patient                 |
| ChestPainType  | Type of chest pain                    |
| RestingBP      | Resting blood pressure                |
| Cholesterol    | Cholesterol level                     |
| FastingBS      | Fasting blood sugar                   |
| RestingECG     | Resting electrocardiogram result      |
| MaxHR          | Maximum heart rate                    |
| ExerciseAngina | Exercise-induced angina               |
| Oldpeak        | ST depression value                   |
| ST_Slope       | Slope of the peak exercise ST segment |
| HeartDisease   | Target variable                       |

The target variable is:

```text
HeartDisease
```

where:

```text
0 → No Heart Disease
1 → Heart Disease
```

The notebook confirms that all 918 records contain no missing values and no duplicate rows.

---

## 🔎 Exploratory Data Analysis

The project performs exploratory data analysis using:

* Dataset shape analysis
* Data type inspection
* Statistical summary
* Missing-value analysis
* Duplicate-value analysis
* Target distribution visualization
* Feature inspection

The dataset contains:

* **918 rows**
* **12 original columns**
* **0 missing values**
* **0 duplicate records**

---

## ⚙️ Data Preprocessing

### 1. Categorical Encoding

Categorical variables are converted into numerical variables using:

```python
pd.get_dummies(data, drop_first=True)
```

This produces encoded features such as:

```text
Sex_M
ChestPainType_ATA
ChestPainType_NAP
ChestPainType_TA
RestingECG_Normal
RestingECG_ST
ExerciseAngina_Y
ST_Slope_Flat
ST_Slope_Up
```

After encoding, the dataset contains **16 columns**, including the target variable.

### 2. Feature Scaling

`StandardScaler` is used to standardize the numerical features:

```python
Age
RestingBP
Cholesterol
MaxHR
Oldpeak
```

The features are transformed to a standardized scale before training the models.

---

## 🤖 Machine Learning Models

Five classification algorithms were tested:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree
5. Support Vector Machine (RBF Kernel)

The dataset was divided into training and testing sets using an **80/20 split** with `random_state=42`.

---

## 📈 Model Performance

The models were evaluated using:

* Accuracy
* F1 Score

| Model                   |   Accuracy |   F1 Score |
| ----------------------- | ---------: | ---------: |
| **Logistic Regression** | **86.41%** | **88.04%** |
| KNN                     |     85.33% |     87.08% |
| Naive Bayes             |     85.33% |     86.83% |
| SVM (RBF Kernel)        |     84.78% |     86.79% |
| Decision Tree           |     76.63% |     79.43% |

Based on the results from the notebook, **Logistic Regression performed best** among the tested models.

---

## 🏆 Final Model

The final model selected for deployment is:

```text
Logistic Regression
```

The trained model is saved using **Joblib**:

```python
joblib.dump(
    models['Logistic Regression'],
    'LOR_heart.pkl'
)
```

The scaler and feature-column information are also saved:

```python
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(x.columns.tolist(), 'columns.pkl')
```

---

## 🌐 Streamlit Application

The trained model is integrated into a Streamlit web application.

Users can enter:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise Angina
* Oldpeak
* ST Slope

The application processes the input using the same feature structure and scaler used during model development and returns a heart disease prediction.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 📂 Project Structure

```text
Heart disease prediction using machine learning, model comparison, Logistic Regression, and Streamlit./
│
├── app.py
│
├── code.ipynb
│
├── LOR_heart.pkl
├── scaler.pkl
├── columns.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File               | Purpose                                         |
| ------------------ | ----------------------------------------------- |
| `app.py`           | Streamlit frontend and prediction application   |
| `code.ipynb`       | Data analysis, preprocessing and model training |
| `LOR_heart.pkl`    | Trained Logistic Regression model               |
| `scaler.pkl`       | Saved feature scaler                            |
| `columns.pkl`      | Saved model feature columns                     |
| `requirements.txt` | Required Python libraries                       |
| `README.md`        | Project documentation                           |
| `.gitignore`       | Files excluded from Git                         |

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
```

### 2. Navigate to the project

```bash
cd heart-disease-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux/macOS

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run Streamlit

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
EDA
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Train Multiple Models
   ↓
Compare Accuracy & F1 Score
   ↓
Select Logistic Regression
   ↓
Save Model using Joblib
   ↓
Streamlit Application
   ↓
Heart Disease Prediction
```

---

## 🎯 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Exploratory Data Analysis
* Data preprocessing
* Categorical feature encoding
* Feature scaling
* Train-test splitting
* Binary classification
* Model comparison
* Accuracy and F1-score evaluation
* Model serialization using Joblib
* Machine learning deployment with Streamlit

---

## 🔮 Future Improvements

Possible improvements to the project include:

* Hyperparameter tuning
* Cross-validation
* Improved preprocessing pipeline
* Model explainability
* Feature importance visualization
* Confusion matrix and ROC-AUC analysis
* Better Streamlit UI/UX
* Cloud deployment
* Automated model retraining

---

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes only**.

The predictions generated by this application should **not be considered a medical diagnosis** and should not replace advice from a qualified healthcare professional.

---

## 👨‍💻 Author

**Mukesh Saran**

GitHub: `https://github.com/mukeshsaran07

---

## ⭐ Project

If you found this project useful, consider giving the repository a ⭐.
