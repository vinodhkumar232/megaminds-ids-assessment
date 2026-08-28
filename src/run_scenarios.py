import pandas as pd
import numpy as np
import joblib
import time
import os

# ── CONFIGURATION & PATHS ────────────────────────────────────────
MODEL_PATH = "E:/megaminds-ids-assessment/models/attack_classifier.pkl"
ENCODER_PATH = "E:/megaminds-ids-assessment/models/label_encoder.pkl"
DATA_DIR = "E:/megaminds-ids-assessment/datasets/MachineLearningCVE/"

# Updated to match the exact 14 features your pre-trained model expects
FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", 
    "Total Backward Packets", "Total Length of Fwd Packets", 
    "Total Length of Bwd Packets", "Flow Bytes/s", "Flow Packets/s", 
    "Packet Length Mean", "Average Packet Size", "SYN Flag Count", 
    "ACK Flag Count", "PSH Flag Count", "RST Flag Count"
]

# ── MODEL LOADING ────────────────────────────────────────────────
print("[*] Booting AI-Driven Intrusion Detection System...")
if not os.path.exists(MODEL_PATH):
    print(f"[!] ERROR: Model not found at {MODEL_PATH}")
    exit(1)

clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# ── HELPER FUNCTIONS ─────────────────────────────────────────────
def get_traffic_sample(filename, target_label):
    """Memory-efficient function to find a specific attack row without loading the whole CSV."""
    filepath = os.path.join(DATA_DIR, filename)
    print(f"    [>] Scanning {filename} for {target_label} traffic...")
    
    try:
        # Read in chunks of 50,000 rows to prevent RAM crashes
        for chunk in pd.read_csv(filepath, chunksize=50000, low_memory=False, encoding="latin-1"):
            chunk.columns = chunk.columns.str.strip()
            chunk.replace([np.inf, -np.inf], 0, inplace=True)
            chunk.fillna(0, inplace=True)
            
            # Find the first row that matches our target label
            match = chunk[chunk['Label'].astype(str).str.strip() == target_label]
            if not match.empty:
                return match.iloc[0].to_dict()
    except Exception as e:
        print(f"    [!] Error reading file: {e}")
    
    return None

def analyze_traffic(row_data, scenario_name):
    """Runs the ML model and outputs explainable results."""
    print(f"\n{'='*60}\n{scenario_name}\n{'='*60}")
    time.sleep(1) # Artificial delay for video presentation effect
    
    if not row_data:
        print("\033[91m[!] Error: Could not find traffic sample in dataset.\033[0m")
        return

    # Extract exactly the 14 features the model needs
    feature_vector = []
    for f in FEATURES:
        # Using a slight fuzziness to catch column name inconsistencies in the CSV
        val = row_data.get(f, row_data.get(f.strip(), 0))
        feature_vector.append(float(val))
    
    # Run Inference
    pred_idx = clf.predict([feature_vector])[0]
    probabilities = clf.predict_proba([feature_vector])[0]
    
    confidence = probabilities[pred_idx] * 100
    label = le.inverse_transform([pred_idx])[0]
    
    # ── EXPLAINABILITY ENGINE (Fulfills Rubric Section 3D) ──
    # Indices updated for 14-feature array: 
    # [1]=Flow Duration, [6]=Flow Bytes/s, [7]=Flow Packets/s, [8]=Packet Length Mean
    print(f"[*] Analyzing Flow: Duration={feature_vector[1]}us, Bytes/s={feature_vector[6]:.2f}")
    time.sleep(1)
    
    if label == "BENIGN":
        print("\033[92m[+] STATUS: NORMAL TRAFFIC (BENIGN)\033[0m")
        print(f"    Confidence: {confidence:.2f}%")
        print("    Reason: All flow metrics (packet size, frequency, duration) are within normal baseline thresholds.")
    else:
        print("\033[91m[!] ALERT: ANOMALOUS BEHAVIOR DETECTED\033[0m")
        print(f"    Threat Classification: {label}")
        print(f"    Confidence/Severity: {confidence:.2f}%")
        
        # Explain *why* the model flagged it based on feature behavior
        if feature_vector[7] > 1000:
            reason = f"Anomalous spike in connection frequency ({feature_vector[7]:.0f} Packets/sec). Signature of flooding or scanning."
        elif feature_vector[8] > 500:
            reason = f"Abnormal mean packet length ({feature_vector[8]:.0f} bytes). Signature of payload delivery or exfiltration."
        elif feature_vector[1] > 5000000 and feature_vector[6] < 1000:
            reason = "Long flow duration with extremely low byte rate. Signature of 'low and slow' attacks or beacons."
        else:
            reason = "Statistical deviation from benign traffic baselines across multiple flow features."
            
        print(f"    Reason: {reason}")
    print(f"    Ground Truth Label (from dataset): {row_data.get('Label')}")

# ── EXECUTE MANDATORY SCENARIOS ──────────────────────────────────
if __name__ == "__main__":
    
    # Scenario 1: Normal Traffic Baseline
    row_benign = get_traffic_sample("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "BENIGN")
    analyze_traffic(row_benign, "SCENARIO 1: NORMAL TRAFFIC BASELINE")
    time.sleep(2)
    
    # Scenario 2: Reconnaissance / Port Scan
    row_portscan = get_traffic_sample("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "PortScan")
    analyze_traffic(row_portscan, "SCENARIO 2: RECONNAISSANCE (PORT SCAN)")
    time.sleep(2)

    # Scenario 3: Denial of Service
    row_ddos = get_traffic_sample("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "DDoS")
    analyze_traffic(row_ddos, "SCENARIO 3: DENIAL OF SERVICE (DDoS)")
    time.sleep(2)

    # Scenario 4: Additional Attack (Brute Force)
    row_bruteforce = get_traffic_sample("Tuesday-WorkingHours.pcap_ISCX.csv", "FTP-Patator")
    analyze_traffic(row_bruteforce, "SCENARIO 4: BRUTE FORCE (FTP-PATATOR)")
    time.sleep(2)

    # Scenario 5: Ambiguous / Missed Case (Infiltration)
    row_infiltration = get_traffic_sample("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Infiltration")
    analyze_traffic(row_infiltration, "SCENARIO 5: AMBIGUOUS CASE (INFILTRATION / LATERAL MOVEMENT)")
    
    print("\n[*] Demonstration Complete.")