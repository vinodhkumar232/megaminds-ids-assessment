import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ── CONFIGURATION ─────────────────────────────────────────────
DATA_PATH = "datasets/MachineLearningCVE/combined_traffic.csv"
MODEL_DIR = "models"

FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Flow Bytes/s", "Flow Packets/s",
    "Packet Length Mean", "Average Packet Size", "SYN Flag Count",
    "ACK Flag Count", "PSH Flag Count", "RST Flag Count",
]

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("[*] Loading combined dataset into memory...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"[!] Error: {DATA_PATH} not found. Run merge_datasets.py first.")
        return
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    print("\n" + "="*70)
    print("STAGE 1: TRAINING UNSUPERVISED BEHAVIORAL BASELINE")
    print("="*70)
    # We strictly isolate Benign traffic to build a mathematical baseline
    # This directly addresses the requirement for genuine anomaly detection
    benign_traffic = df[df["Label"] == "BENIGN"][FEATURES]
    
    # We sample 200k flows to prevent memory exhaustion and speed up training. 
    # 200k is more than enough to establish a dense statistical baseline.
    sample_size = min(200000, len(benign_traffic))
    baseline_sample = benign_traffic.sample(n=sample_size, random_state=42)
    
    print(f"[*] Training Isolation Forest on {sample_size:,} benign flows...")
    # contamination=0.01 allows for 1% of our labeled "benign" data to actually be hidden noise
    iso_forest = IsolationForest(n_estimators=100, max_samples=256, contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(baseline_sample)
    
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "anomaly_detector.pkl"))
    print(f"[+] Baseline Engine saved to {MODEL_DIR}/anomaly_detector.pkl")
    
    print("\n" + "="*70)
    print("STAGE 2: TRAINING SUPERVISED SIGNATURE CLASSIFIER")
    print("="*70)
    
    X = df[FEATURES]
    y_labels = df["Label"]
    
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    
    print("[*] Executing 70/30 stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    
    print(f"[*] Training Random Forest on {len(X_train):,} total flows...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    joblib.dump(clf, os.path.join(MODEL_DIR, "attack_classifier.pkl"))
    print(f"[+] Signature Engine saved to {MODEL_DIR}/attack_classifier.pkl")
    
    print("\n[*] Evaluating Supervised Model on Holdout Set...")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
    
    with open(os.path.join(MODEL_DIR, "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("HYBRID MODEL EVALUATION: SUPERVISED COMPONENT\n")
        f.write("="*60 + "\n")
        f.write(report)
        
    print("[+] Evaluation metrics exported successfully.")
    print("[*] Hybrid Training Pipeline Complete.")

if __name__ == "__main__":
    main()