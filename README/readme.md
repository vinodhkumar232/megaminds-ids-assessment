# AI-Driven Network Intrusion Detection System

An enterprise-grade, machine learning-based Intrusion Detection System (IDS) designed to identify and classify complex network attacks. This pipeline ingests network telemetry, maintains rolling traffic baselines, and flags anomalous campaigns using a highly optimized Random Forest classifier.

## Key Capabilities
* **Streaming Ingestion Engine:** Processes network flows in sequential batches with memory-safe `deque` buffering, realistically simulating live network traffic monitoring rather than relying on static single-row lookups.
* **Multi-Class Threat Detection:** Trained on the complete CIC-IDS-2017 dataset to identify 15 distinct traffic profiles (Benign + 14 attack vectors including DDoS, PortScan, FTP-Patator, and Infiltration).
* **True ML Explainability:** Utilizes dynamic feature importance weighting directly from the Random Forest model to mathematically explain *why* a specific flow was flagged (e.g., extracting exact anomalous packet rates or duration thresholds).
* **High-Fidelity Classification:** Achieves **98.41% overall accuracy** across 2.8 million flows using a synchronized 14-feature vector and balanced class weighting.

## Repository Structure
* `models/` - Contains the trained `.pkl` model artifacts, label encoders, and exported evaluation metrics.
* `src/` - Core Python pipeline (dataset merging, training, and real-time inference scenarios).
* `datasets/MachineLearningCVE/` - Directory for the raw CIC-IDS-2017 CSV files (excluded via `.gitignore` due to size constraints).
* `Technical_Report.pdf` - Comprehensive technical documentation, architecture overview, and evaluation results.
* `requirements.txt` - Python environment dependencies.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vinodhkumar232/megaminds-ids-assessment.git](https://github.com/vinodhkumar232/megaminds-ids-assessment.git)
   cd megaminds-ids-assessment
