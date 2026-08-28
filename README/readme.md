# AI-Driven Network Traffic Analysis and Intrusion Detection System

## Project Overview
This repository contains a working prototype of an AI-driven Intrusion Detection System (IDS). Unlike traditional signature-based systems (e.g., Snort), this system utilizes a machine learning approach (Random Forest Classifier) to analyze network traffic behavior and detect anomalous patterns such as DDoS, Port Scanning, Brute Force, and Infiltration attempts.

## Repository Structure
* `/src/run_scenarios.py`: The Command-Line Interface (CLI) that ingests PCAP CSVs, extracts features, runs inference, and provides explainable alert outputs.
* `/models/attack_classifier.pkl`: The trained Random Forest model.
* `/models/label_encoder.pkl`: The label encoder mapping integer predictions to threat classifications.
* `/data/`: Contains sample `.csv` files from the CIC-IDS-2017 dataset used for scenario testing.
* `Technical_Report.pdf`: Comprehensive documentation covering architecture, methodology, and evaluation.

## Installation & Setup
1. Clone this repository to your local machine.
2. Ensure Python 3.8+ is installed.
3. Install the required dependencies:
   `pip install pandas numpy scikit-learn joblib`

## Traffic Data (Dataset)
This project utilizes the publicly available **CIC-IDS-2017** dataset, which contains benign and up-to-date common attacks. 
* To reproduce the exact results, download the MachineLearningCVE CSV files from the University of New Brunswick dataset portal.
* Place the downloaded `.csv` files inside the `datasets/MachineLearningCVE/` directory.

## Execution Instructions
To run the 5 mandatory detection scenarios (Normal, PortScan, DDoS, Brute Force, and Ambiguous Infiltration), execute the following command from the project root:

```bash
python src/run_scenarios.py