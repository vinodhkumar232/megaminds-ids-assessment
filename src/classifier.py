import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── CONFIGURATION & CONSTANTS ────────────────────────────────────
DATASET_PATH = "datasets/MachineLearningCVE/combined_traffic.csv"
MODEL_DIR = "models"

# 14 features matching the saved model specification
FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Average Packet Size",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "RST Flag Count",
]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Strips column whitespace and cleans invalid numeric values."""
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def train_model():
    print("[*] Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Error: Dataset not found at {DATASET_PATH}")
        print(
            "    Run 'python src/merge_datasets.py' first to generate combined_traffic.csv"
        )
        return

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    print("[*] Sanitizing dataset (stripping whitespace, removing NaNs/Infs)...")
    df = clean_dataset(df)

    # Check for missing feature columns
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        print(f"[!] Error: Dataset is missing required columns: {missing_cols}")
        return

    X = df[FEATURES]
    y = df["Label"].astype(str).str.strip()

    print(f"[*] Total valid flows loaded: {len(df):,}")
    print(f"[*] Distinct target classes ({len(y.unique())}): {list(y.unique())}")

    print("[*] Encoding target labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("[*] Splitting dataset (70% Training / 30% Testing)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    print(
        "[*] Training Random Forest Classifier (n_estimators=100, max_depth=20, class_weight='balanced_subsample')..."
    )
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("[*] Evaluating model on unseen holdout test set (30%)...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, digits=4, zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 60)
    print(report)

    # Persist evaluation metrics for technical report verification
    os.makedirs(MODEL_DIR, exist_ok=True)
    metrics_path = os.path.join(MODEL_DIR, "evaluation_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {accuracy * 100:.4f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"[+] Saved evaluation metrics to {metrics_path}")

    # Export synchronized model artifacts
    print(f"[*] Saving model artifacts into '{MODEL_DIR}/'...")
    joblib.dump(clf, os.path.join(MODEL_DIR, "attack_classifier.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_names.pkl"))
    print("[+] Training complete. Pipeline artifacts are fully synchronized.")


if __name__ == "__main__":
    train_model()