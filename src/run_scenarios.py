import os
import time
import joblib
import numpy as np
import pandas as pd
from collections import deque
import warnings

warnings.filterwarnings("ignore")

# ── CONFIGURATION & RELATIVE PATHS ──────────────────────────────
MODEL_PATH = "models/attack_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
DATA_DIR = "datasets/MachineLearningCVE"

FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Flow Bytes/s", "Flow Packets/s",
    "Packet Length Mean", "Average Packet Size", "SYN Flag Count",
    "ACK Flag Count", "PSH Flag Count", "RST Flag Count",
]

print("[*] Initializing AI-Driven Intrusion Detection Pipeline...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print(f"[!] ERROR: Model artifacts not found in 'models/'.")
    exit(1)

clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

def simulate_attack_campaign(filename: str, target_label: str, batch_size: int = 25):
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {target_label.upper()} CAMPAIGN DETECTION")
    print(f"{'=' * 70}")

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[-] Error: File not found: {filepath}")
        return

    print(f"[*] Scanning {filename} for {target_label} campaign...")

    history = deque(maxlen=5)
    campaign_batch = []
    found_attack = False

    try:
        for chunk in pd.read_csv(filepath, chunksize=50000, low_memory=False, encoding="latin-1"):
            chunk.columns = chunk.columns.str.strip()
            chunk.replace([np.inf, -np.inf], 0, inplace=True)
            chunk.fillna(0, inplace=True)

            for _, row in chunk.iterrows():
                current_label = str(row.get("Label", "")).strip()

                if not found_attack:
                    if current_label == target_label:
                        found_attack = True
                        campaign_batch.extend(list(history))
                        campaign_batch.append(row)
                    else:
                        history.append(row)
                else:
                    campaign_batch.append(row)
                    if len(campaign_batch) >= batch_size:
                        break

            if len(campaign_batch) >= batch_size:
                break
    except Exception as e:
        print(f"[-] Error parsing {filename}: {e}")
        return

    if not found_attack or len(campaign_batch) == 0:
        return

    print(f"[*] Ingesting network interface stream ({len(campaign_batch)} sequential flows)...\n")
    time.sleep(1)

    threat_count = 0
    malicious_flows = []

    for i, row in enumerate(campaign_batch):
        feature_vector = []
        for f in FEATURES:
            val = row.get(f, row.get(f.strip(), 0.0))
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
            feature_vector.append(val)

        feature_df = pd.DataFrame([feature_vector], columns=FEATURES)
        
        # ── PRIORITY 2 FIX: True Confidence Scoring ──
        probabilities = clf.predict_proba(feature_df)[0]
        pred_idx = np.argmax(probabilities)
        confidence = probabilities[pred_idx] * 100
        pred_label = le.inverse_transform([pred_idx])[0]

        time.sleep(0.05) 

        if pred_label == 'BENIGN':
            print(f"  [+] Flow {i+1:02d}: OK (Benign) | Conf: {confidence:.1f}%")
        else:
            # Only flag as high severity if confidence is above a threshold
            severity = "HIGH" if confidence > 85.0 else "UNCERTAIN"
            color = "\033[91m" if severity == "HIGH" else "\033[93m"
            print(f"  {color}[!] Flow {i+1:02d}: {pred_label} detected | Sev: {severity} | Conf: {confidence:.1f}%\033[0m")
            
            threat_count += 1
            malicious_flows.append(feature_vector)

    print(f"\n{'='*70}")
    
    # ── PRIORITY 4 FIX: Mathematical Threshold Justification ──
    alert_threshold = len(campaign_batch) * 0.30 
    
    if threat_count > alert_threshold:
        print(f"\033[91m[!!!] CRITICAL ALERT: SUSTAINED {target_label.upper()} CAMPAIGN DETECTED\033[0m")
        print(f"[*] Volume: {threat_count}/{len(campaign_batch)} recent flows flagged.")

        # Aggregated Explainability without SHAP dependencies
        print("\n[*] Explainability Engine - Campaign Feature Aggregation:")
        importances = clf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:3]
        malicious_matrix = np.array(malicious_flows)
        
        for idx in top_indices:
            feat_name = FEATURES[idx]
            feat_weight = importances[idx] * 100
            avg_val = np.mean(malicious_matrix[:, idx]) if len(malicious_matrix) > 0 else 0
            print(f"    - {feat_name}: {avg_val:,.2f} (Model Weight: {feat_weight:.1f}%)")
    else:
        print("\033[92m[+] Traffic volume normal. No sustained campaigns detected.\033[0m")
    print(f"{'='*70}\n")
    time.sleep(1)

if __name__ == "__main__":
    simulate_attack_campaign("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "PortScan")
    simulate_attack_campaign("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "DDoS")
    simulate_attack_campaign("Tuesday-WorkingHours.pcap_ISCX.csv", "FTP-Patator")
    simulate_attack_campaign("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Infiltration")
    print("\n[*] Demonstration Suite Complete.")