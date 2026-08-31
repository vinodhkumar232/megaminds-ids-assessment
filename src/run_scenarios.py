import os
import time
import joblib
import numpy as np
import pandas as pd
from collections import deque
import warnings

warnings.filterwarnings("ignore")

# ── CONFIGURATION & RELATIVE PATHS ──────────────────────────────
RF_MODEL_PATH = "models/attack_classifier.pkl"
IF_MODEL_PATH = "models/anomaly_detector.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
DATA_DIR = "datasets/MachineLearningCVE"

FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Flow Bytes/s", "Flow Packets/s",
    "Packet Length Mean", "Average Packet Size", "SYN Flag Count",
    "ACK Flag Count", "PSH Flag Count", "RST Flag Count",
]

print("[*] Initializing Hybrid AI-Driven Intrusion Detection Pipeline...")
if not all(os.path.exists(p) for p in [RF_MODEL_PATH, IF_MODEL_PATH, ENCODER_PATH]):
    print(f"[!] ERROR: Model artifacts not found. Run classifier.py first.")
    exit(1)

clf = joblib.load(RF_MODEL_PATH)
iso_forest = joblib.load(IF_MODEL_PATH)
le = joblib.load(ENCODER_PATH)

def run_blind_inference_scenario(filename: str, target_label: str, window_size: int = 30):
    print(f"\n{'=' * 90}")
    print(f"SCENARIO: BLIND INFERENCE STREAM ({target_label.upper()} ENVIRONMENT)")
    print(f"{'=' * 90}")

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[-] Error: File not found: {filepath}")
        return

    print(f"[*] Fast-forwarding stream to active threat window for demonstration...")

    # Fast-forward buffer to find a mixed window of benign and attack traffic
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
                    if len(campaign_batch) >= window_size:
                        break

            if len(campaign_batch) >= window_size:
                break
    except Exception as e:
        print(f"[-] Error parsing {filename}: {e}")
        return

    if len(campaign_batch) == 0:
        return

    print(f"[*] Ingesting {len(campaign_batch)} blind flows. Ground truth hidden from models...\n")
    time.sleep(1)

    threat_count = 0
    malicious_flows = []

    # Print Table Header
    print(f"{'PREDICTION':<25} | {'CONFIDENCE':<10} | {'BEHAVIORAL BASELINE':<20} || {'ACTUAL GROUND TRUTH'}")
    print("-" * 90)

    for i, row in enumerate(campaign_batch):
        # 1. Extract true label (hidden from engine)
        actual_label = str(row.get("Label", "")).strip()
        
        # 2. Extract strictly the 14 features
        feature_vector = []
        for f in FEATURES:
            try:
                val = float(row.get(f, row.get(f.strip(), 0.0)))
            except (ValueError, TypeError):
                val = 0.0
            feature_vector.append(val)

        feature_df = pd.DataFrame([feature_vector], columns=FEATURES)
        
        # ── HYBRID ENGINE LOGIC ──
        # A. Unsupervised Anomaly Detection (Isolation Forest)
        if_pred = iso_forest.predict(feature_df)[0]
        baseline_status = "NORMAL" if if_pred == 1 else "ANOMALY DEVIATION"

        # B. Supervised Signature Classification (Random Forest)
        probabilities = clf.predict_proba(feature_df)[0]
        pred_idx = np.argmax(probabilities)
        confidence = probabilities[pred_idx] * 100
        pred_label = le.inverse_transform([pred_idx])[0]

        time.sleep(0.05) 

        # Format the output to show Prediction vs Actual
        if pred_label == 'BENIGN' and baseline_status == 'NORMAL':
            print(f"  \033[92m{pred_label:<23}\033[0m | {confidence:>5.1f}%     | {baseline_status:<18} || {actual_label}")
        else:
            threat_count += 1
            malicious_flows.append(feature_vector)
            
            # Dynamic severity logic
            sev_color = "\033[93m" if confidence < 85.0 else "\033[91m"
                
            print(f"  {sev_color}{pred_label:<23}\033[0m | {sev_color}{confidence:>5.1f}%\033[0m     | {sev_color}{baseline_status:<18}\033[0m || {actual_label}")

    print(f"\n{'='*90}")
    
    # ── CAMPAIGN THRESHOLD & EXPLAINABILITY ──
    alert_threshold = len(campaign_batch) * 0.30 
    
    if threat_count > alert_threshold:
        print(f"\033[91m[!!!] CRITICAL ALERT: SUSTAINED CAMPAIGN DETECTED\033[0m")
        print(f"[*] Volume: {threat_count}/{len(campaign_batch)} flows deviated from normal baseline.")

        print("\n[*] Explainability Engine - Campaign Feature Aggregation:")
        importances = clf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:3]
        malicious_matrix = np.array(malicious_flows)
        
        for idx in top_indices:
            feat_name = FEATURES[idx]
            feat_weight = importances[idx] * 100
            avg_val = np.mean(malicious_matrix[:, idx]) if len(malicious_matrix) > 0 else 0
            print(f"    - {feat_name}: {avg_val:,.2f} (RF Model Weight: {feat_weight:.1f}%)")
    else:
        print("\033[92m[+] Traffic volume normal. No sustained campaigns detected.\033[0m")
    print(f"{'='*90}\n")
    time.sleep(1)

if __name__ == "__main__":
    run_blind_inference_scenario("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "PortScan")
    run_blind_inference_scenario("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "DDoS")
    run_blind_inference_scenario("Tuesday-WorkingHours.pcap_ISCX.csv", "FTP-Patator")
    run_blind_inference_scenario("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Infiltration")
    print("\n[*] Hybrid Engine Demonstration Suite Complete.")