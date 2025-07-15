import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("heart.csv")
df.columns = df.columns.str.strip()
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

# Accuracy
print("Model Accuracy on Test Set: {:.2f}%".format(model.score(X_test, y_test) * 100))

# Input prompt
print("\nEnter patient details:")
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

# Encode user inputs using the same mapping as dataset
sex_map = {'M': 1, 'F': 0}
cp_map = {'ATA': 1, 'NAP': 2, 'ASY': 0, 'TA': 3}
restecg_map = {'NORMAL': 1, 'ST': 2, 'T': 0}
exang_map = {'Y': 1, 'N': 0}
slope_map = {'UP': 2, 'FLAT': 1, 'DOWN': 0}

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

# Output
print("\nHeart Attack Risk Prediction:")
print("Risk Level:", "High 🔴" if risk == 1 else "Low 🟢")
