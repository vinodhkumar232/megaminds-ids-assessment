import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ── CONFIGURATION ────────────────────────────────────────────────
CSV_PATH = "E:/rawlogs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# Exact features extracted by your data_loader.py
FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Fwd Packets/s",
    "Packet Length Mean"
]

def clean_dataset(df):
    """Handles infinite and NaN values common in CIC-IDS-2017."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def train_model():
    print(f"[*] Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Strip whitespace from column names (critical for CIC-IDS-2017)
    df.columns = df.columns.str.strip()
    
    print("[*] Sanitizing data...")
    df = clean_dataset(df)
    
    X = df[FEATURES]
    y = df["Label"]
    
    # Encode text labels into integers for the model
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("[*] Splitting data into 70% Training and 30% Testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    
    print("[*] Training Random Forest Classifier (this may take a minute)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print("\n" + "="*50)
    print("  MODEL EVALUATION METRICS (For Technical Report)")
    print("="*50)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\n" + "="*50)
    print("  FEATURE IMPORTANCE (For Explainability Requirement)")
    print("="*50)
    importances = clf.feature_importances_
    for feature, importance in sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True):
        print(f"{feature:>25}: {importance:.4f}")
        
    print("\n[*] Exporting model and encoder for dataset_uploader.py...")
    joblib.dump(clf, "attack_classifier.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("[+] Export complete. System is ready.")

def predict_attack(raw_row):
    """
    Called dynamically by dataset_uploader.py to classify live/uploaded data.
    Takes a dictionary of network metrics, returns (Attack Label, Confidence %).
    """
    if not os.path.exists("attack_classifier.pkl"):
        raise FileNotFoundError("Model not trained. Run classifier.py first.")
        
    clf = joblib.load("attack_classifier.pkl")
    le = joblib.load("label_encoder.pkl")
    
    # Extract only the features the model was trained on
    clean_raw = {str(k).strip().lower(): v for k, v in raw_row.items()}
    
    feature_vector = []
    for feature in FEATURES:
        # Match incoming dict keys (case-insensitive) to required features
        val = clean_raw.get(feature.lower(), 0)
        feature_vector.append(float(val) if val not in ["inf", "Inf", "nan", None] else 0.0)
        
    # Predict
    pred_idx = clf.predict([feature_vector])[0]
    probabilities = clf.predict_proba([feature_vector])[0]
    
    confidence = round(probabilities[pred_idx] * 100, 2)
    predicted_label = le.inverse_transform([pred_idx])[0]
    
    return predicted_label, confidence

if __name__ == "__main__":
    train_model()