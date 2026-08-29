import pandas as pd
import numpy as np

# Point this to your actual combined CSV
CSV_PATH = "datasets/MachineLearningCVE/combined_traffic.csv"

FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", 
    "Total Backward Packets", "Total Length of Fwd Packets", 
    "Total Length of Bwd Packets", "Flow Bytes/s", "Flow Packets/s", 
    "Packet Length Mean", "Average Packet Size", "SYN Flag Count", 
    "ACK Flag Count", "PSH Flag Count", "RST Flag Count"
]

def verify_dataset():
    print(f"[*] Loading {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("[-] FATAL: File not found.")
        return

    # 1. Check for missing columns
    missing_cols = [col for col in FEATURES + ['Label'] if col not in df.columns]
    if missing_cols:
        print(f"[-] FATAL: Missing columns: {missing_cols}")
        return
    print("[+] All required feature columns are present.")

    # 2. Check for NaNs and Infinite values
    df_features = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    if df_features.isna().any().any():
        print("[-] WARNING: Dataset contains NaNs or Infinite values.")
        print("    You must add dropna() or fillna() logic to your inference script.")
    else:
        print("[+] Dataset is perfectly clean (No NaNs/Infs).")

    # 3. List available scenarios
    print("\n[+] Available attack campaigns for simulation:")
    print(df['Label'].value_counts())

if __name__ == "__main__":
    verify_dataset()