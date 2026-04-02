import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
from utils import ALL_FEATURE_NAMES

warnings.filterwarnings('ignore')

def extract_meaningful_shaps():
    print("Extracting Explainability matrix from raw Tree model...")
    data_path = 'data/processed_tox21.csv'
    model_path = 'data/best_model_raw.pkl' 
    
    if not os.path.exists(data_path):
        return
        
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    
    # Fast rendering by sampling strictly bound limits
    X = df[ALL_FEATURE_NAMES].sample(500, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X)
    
    if isinstance(shap_values_raw, list) and len(shap_values_raw) > 1:
        shap_values = shap_values_raw[1]
    else:
        shap_values = shap_values_raw[:, :, 1] if len(shap_values_raw.shape) == 3 else shap_values_raw
        
    explainer_obj = explainer(X)
    if len(explainer_obj.values.shape) == 3:
        explainer_obj = explainer_obj[:, :, 1]
    
    os.makedirs('data', exist_ok=True)
    
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig('data/shap_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=10)
    plt.tight_layout()
    plt.savefig('data/shap_bar.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure()
    shap.waterfall_plot(explainer_obj[0], show=False, max_display=10)
    plt.tight_layout()
    plt.savefig('data/shap_waterfall.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Saved SHAP visuals. Output safely exported.")

if __name__ == '__main__':
    extract_meaningful_shaps()
