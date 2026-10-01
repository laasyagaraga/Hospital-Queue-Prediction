"""Train the hospital waiting-time demo model from dataset.csv.
The included dataset is synthetic demonstration data, not real hospital records.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "dataset.csv"
MODEL_PATH = BASE / "hospital_model.pkl"
ENCODERS_PATH = BASE / "label_encoders.pkl"

CATEGORICAL = ["Hospital", "Department", "Day", "Holiday"]
FEATURES = ["Hospital", "Department", "Day", "Time", "Patients", "DoctorsAvailable", "EmergencyCases", "Holiday"]

def main():
    data = pd.read_csv(DATA_PATH)
    encoders = {}
    encoded = data.copy()
    for col in CATEGORICAL:
        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(data[col].astype(str))
        encoders[col.lower()] = encoder
    X = encoded[FEATURES]
    y = encoded["WaitTime"]
    model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"Model trained successfully using {len(data)} synthetic demo rows.")
    print(f"Saved model to {MODEL_PATH.name} and encoders to {ENCODERS_PATH.name}.")

if __name__ == "__main__":
    main()
