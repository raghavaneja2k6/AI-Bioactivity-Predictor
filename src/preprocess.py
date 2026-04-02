import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
import warnings

# Use the centralized modular logic natively requested
from utils import feature_extraction, DESCRIPTOR_NAMES, FP_NAMES, ALL_FEATURE_NAMES

warnings.filterwarnings('ignore')

def preprocess_features():
    print("--- Step 1: Loading Raw Data ---")
    input_path = 'data/tox21.csv'
    public_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
    output_path = 'data/processed_tox21.csv'
    scaler_path = 'data/scaler.pkl'
    
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"Local {input_path} not found. Fetching real-world dataset from DeepChem Public S3...")
        try:
            df = pd.read_csv(public_url)
            df.to_csv(input_path, index=False)
            print(f"Successfully downloaded Tox21 dataset with shape: {df.shape}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return
    else:
        df = pd.read_csv(input_path)
        
    print(f"Operating Shape: {df.shape}")
    
    # ----------------------------------------------------
    # Target Construction
    # ----------------------------------------------------
    df = df.dropna(subset=['NR-AR'])
    df['toxicity'] = df['NR-AR'].astype(int)
    print("Created unified binary 'toxicity' column targeting 'NR-AR'.")
    
    # Clean duplicates based on smiles structure
    initial_len = len(df)
    df = df.drop_duplicates(subset=['smiles'])
    print(f"Removed {initial_len - len(df)} duplicate SMILES.")
    
    # ----------------------------------------------------
    # 2056-Feature Extraction Integration
    # ----------------------------------------------------
    print("--- Step 2: 2056-Feature Extraction Matrix (RDKit) ---")
    # Apply standard modular function row-wise natively
    feature_matrix = df['smiles'].apply(lambda x: pd.Series(feature_extraction(x)))
    feature_matrix.columns = ALL_FEATURE_NAMES
    
    # Concatenate features into DataFrame natively
    df = pd.concat([df, feature_matrix], axis=1)
    
    # Drop rows that failed SMILES topology evaluation mapping natively
    initial_len = len(df)
    df = df.dropna(subset=DESCRIPTOR_NAMES)
    print(f"Dropped {initial_len - len(df)} rows due to invalid SMILES. Remaining Valid: {len(df)}")
    
    # ----------------------------------------------------
    # Safe Numerical Feature Scaling
    # ----------------------------------------------------
    print("--- Step 3: Partial Scaling & Matrix Alignment ---")
    print(f"Scaling exactly {len(DESCRIPTOR_NAMES)} pure descriptors... (Leaving {len(FP_NAMES)} fingerprints untouched)")
    scaler = StandardScaler()
    df[DESCRIPTOR_NAMES] = scaler.fit_transform(df[DESCRIPTOR_NAMES])
    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    # Data summary readout
    y = df['toxicity']
    print(f"Toxic: {(y==1).sum()} | Safe: {(y==0).sum()}")
    
    final_cols = ['smiles'] + ALL_FEATURE_NAMES + ['toxicity']
    df[final_cols].to_csv(output_path, index=False)
    print(f"--- Pipeline Finished. Successfully saved to {output_path} ---")

if __name__ == '__main__':
    preprocess_features()
