import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("heart.csv")
df.columns = df.columns.str.strip()  # Clean column names
print("Columns in dataset:", df.columns.tolist())

# Encode categorical columns
label_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)

# Accuracy & Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print("Accuracy      : {:.2f}%".format(acc * 100))
print("Precision     : {:.2f}%".format(prec * 100))
print("Recall        : {:.2f}%".format(rec * 100))
print("F1 Score      : {:.2f}%".format(f1 * 100))
print("Confusion Matrix:")
print(cm)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# User Input
print("\n🔍 Enter patient details:")
age = int(input("Age: "))
sex = input("Sex (M/F): ").strip().upper()
cp = input("Chest Pain Type (ATA/NAP/ASY/TA): ").strip().upper()
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false): "))
restecg = input("Resting ECG (Normal/ST/T): ").strip().upper()
thalach = float(input("Maximum Heart Rate: "))
exang = input("Exercise Induced Angina (Y/N): ").strip().upper()
oldpeak = float(input("ST Depression: "))
slope = input("ST Slope (Up/Flat/Down): ").strip().upper()

# Mappings for categorical fields (should match encoded training data)
sex_map = {'M': 1, 'F': 0}
cp_map = {'ATA': 1, 'NAP': 2, 'ASY': 0, 'TA': 3}
restecg_map = {'NORMAL': 1, 'ST': 2, 'T': 0}
exang_map = {'Y': 1, 'N': 0}
slope_map = {'UP': 2, 'FLAT': 1, 'DOWN': 0}

# Input data as DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'Sex': sex_map.get(sex, 1),
    'ChestPainType': cp_map.get(cp, 1),
    'RestingBP': trestbps,
    'Cholesterol': chol,
    'FastingBS': fbs,
    'RestingECG': restecg_map.get(restecg, 1),
    'MaxHR': thalach,
    'ExerciseAngina': exang_map.get(exang, 0),
    'Oldpeak': oldpeak,
    'ST_Slope': slope_map.get(slope, 2)
}])

# Predict
risk = model.predict(input_data)[0]
confidence = model.predict_proba(input_data)[0][1]  # Probability of class 1 (High Risk)

# Output
print("\n❤️ Heart Attack Risk Prediction:")
print("Risk Level    :", "High 🔴" if risk == 1 else "Low 🟢")
print("Confidence    : {:.2f}%".format(confidence * 100))
