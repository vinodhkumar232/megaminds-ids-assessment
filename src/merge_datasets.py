import os
import glob
import pandas as pd
import numpy as np

def merge_and_clean_datasets():
    dataset_dir = "datasets/MachineLearningCVE"
    output_file = os.path.join(dataset_dir, "combined_traffic.csv")
    
    # 14 features + 1 Label
    TARGET_COLUMNS = [
        "Destination Port", "Flow Duration", "Total Fwd Packets", 
        "Total Backward Packets", "Total Length of Fwd Packets", 
        "Total Length of Bwd Packets", "Flow Bytes/s", "Flow Packets/s", 
        "Packet Length Mean", "Average Packet Size", "SYN Flag Count", 
        "ACK Flag Count", "PSH Flag Count", "RST Flag Count", "Label"
    ]

    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    csv_files = [f for f in csv_files if "combined_traffic.csv" not in f]
    
    if not csv_files:
        print(f"[-] No CSV files found in {dataset_dir}")
        return

    print(f"[*] Found {len(csv_files)} CSV files. Starting optimized merge...")
    
    df_list = []
    for file in csv_files:
        print(f"    -> Ingesting {os.path.basename(file)}...")
        try:
            df = pd.read_csv(file, low_memory=False)
            df.columns = df.columns.str.strip()
            
            missing_cols = [col for col in TARGET_COLUMNS if col not in df.columns]
            if missing_cols:
                print(f"       [!] Skipping file. Missing columns: {missing_cols}")
                continue
            
            df_subset = df[TARGET_COLUMNS]
            df_list.append(df_subset)
            
        except Exception as e:
            print(f"       [!] Error reading {file}: {e}")
    
    if not df_list:
        print("[-] No valid dataframes to merge. Check your dataset folder.")
        return
        
    print("\n[*] Concatenating data into a single master dataframe...")
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print(f"[*] Initial combined shape: {combined_df.shape}")
    print("[*] Sanitizing data (Removing NaNs and Infinity values)...")
    
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(inplace=True)
    
    print(f"[*] Final sanitized shape: {combined_df.shape}")
    print("\n[*] Exporting unified dataset to...")
    combined_df.to_csv(output_file, index=False)
    print("[+] Merge complete! Your dataset is 100% synchronized and ready.")

if __name__ == "__main__":
    merge_and_clean_datasets()