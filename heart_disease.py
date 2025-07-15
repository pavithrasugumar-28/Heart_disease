import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Print model accuracy
print("Model Accuracy on Test Set: {:.2f}%".format(model.score(X_test, y_test) * 100))

# User input
print("\nEnter patient details:")
age = int(input("Age: "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
thalach = float(input("Maximum Heart Rate: "))
oldpeak = float(input("ST Depression: "))
cp = int(input("Chest Pain Type (0-3): "))
slope = int(input("Slope of ST Segment (0-2): "))
ca = int(input("Major Vessels Colored by Fluoroscopy (0-3): "))
sex = int(input("Sex (1 = male; 0 = female): "))
fbs = int(input("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false): "))
restecg = int(input("Resting ECG Results (0-2): "))
exang = int(input("Exercise Induced Angina (1 = yes; 0 = no): "))

# Create input DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'trestbps': trestbps,
    'chol': chol,
    'thalach': thalach,
    'oldpeak': oldpeak,
    'cp': cp,
    'slope': slope,
    'ca': ca,
    'sex': sex,
    'fbs': fbs,
    'restecg': restecg,
    'exang': exang
}])

# Predict
risk = model.predict(input_data)[0]

# Output
print("\nHeart Attack Risk Prediction:")
print("Risk Level:", "High 🔴" if risk == 1 else "Low 🟢")
