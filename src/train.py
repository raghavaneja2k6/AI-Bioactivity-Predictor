import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import ALL_FEATURE_NAMES
from imblearn.over_sampling import SMOTE

def train_model():
    filepath = 'data/processed_tox21.csv'
    if not os.path.exists(filepath):
        print(f"Dataset {filepath} missing!")
        return

    print("Loading 2056-feature matrix natively...")
    df = pd.read_csv(filepath)
    X = df[ALL_FEATURE_NAMES]
    y = df['toxicity']
    
    # 80/20 train/test split natively avoiding data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Computing class distribution natively:")
    print(y_train.value_counts())
    
    print("Increasing mathematical weight of the top 8 structural descriptors...")
    X_train.iloc[:, :8] *= 2.5
    X_test.iloc[:, :8] *= 2.5
    
    print("Applying synthetic SMOTE arrays against imbalanced classifications...")
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
    
    # Required configurations directly
    print("Initializing heavy Random Forest structural configurations...")
    raw_rf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=12, 
        class_weight="balanced", 
        random_state=42,
        n_jobs=-1
    )
    
    # Sigmoid explicitly mandated
    print("Fitting CalibratedClassifierCV(method='sigmoid'...)...")
    calibrated_rf = CalibratedClassifierCV(raw_rf, method='sigmoid', cv=3)
    calibrated_rf.fit(X_resampled, y_resampled)
    
    print("Extracting Test Metrics natively...")
    y_pred = calibrated_rf.predict(X_test)
    y_proba = calibrated_rf.predict_proba(X_test)[:, 1]
    
    print("Evaluation Metrics (Calibrated Test Set):")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    
    print("Exporting dual-class binary storage sequentially...")
    os.makedirs('data', exist_ok=True)
    # Fit the raw model natively securely to save for SHAP
    raw_rf.fit(X_resampled, y_resampled)
    joblib.dump(raw_rf, 'data/best_model_raw.pkl')
    joblib.dump(calibrated_rf, 'data/best_model.pkl')
    
    print("\n--- Sanity Checks ---")
    scaler = joblib.load('data/scaler.pkl')
    from utils import feature_extraction
    
    test_smiles = ['C', 'CCO', 'c1ccccc1', 'CCCCCCCCCCCCCCCC']
    for s in test_smiles:
        feats = feature_extraction(s)
        scaled_desc = scaler.transform(feats[:8].reshape(1, -1))
        # Natively map the identical mathematical weight (2.5) during manual checks to prove continuity
        scaled_desc *= 2.5
        comb = np.concatenate((scaled_desc.flatten(), feats[8:]))
        prob = calibrated_rf.predict_proba(comb.reshape(1, -1))[0, 1]
        print(f"SMILES: {s:<20} -> Toxicity Probability: {prob*100:.1f}%")
        
    print("\nPipeline Execution finished reliably.")

if __name__ == '__main__':
    train_model()
