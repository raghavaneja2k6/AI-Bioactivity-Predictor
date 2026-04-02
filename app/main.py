import math
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import shap
import sys
from PIL import Image

# ---------------- RDKit SAFE ----------------
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False

# ---------------- IMPORT UTILS ----------------
sys.path.append('src')
from utils import feature_extraction, DESCRIPTOR_NAMES, FP_NAMES, ALL_FEATURE_NAMES

st.set_page_config(page_title="AI Bioactivity Predictor (NR-AR)", layout="wide")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("data/best_model.pkl")
        raw_model = joblib.load("data/best_model_raw.pkl")
        scaler = joblib.load("data/scaler.pkl")
        explainer = shap.TreeExplainer(raw_model)
        return model, scaler, explainer
    except:
        return None, None, None

model, scaler, explainer = load_artifacts()

if model is None:
    st.error("Model not loaded")
    st.stop()

# ---------------- UI ----------------
st.title("🧬 AI Bioactivity Predictor (NR-AR)")
st.markdown("Predict androgen receptor activity using molecular features and ML")

smiles = st.text_input("Enter SMILES", "CCO")

# ---------------- FEATURES ----------------
features = feature_extraction(smiles)

if np.isnan(features[0]):
    st.error("Invalid SMILES")
    st.stop()

desc = dict(zip(DESCRIPTOR_NAMES, features[:len(DESCRIPTOR_NAMES)]))
df = pd.DataFrame([desc])
fp = features[len(DESCRIPTOR_NAMES):]

# ---------------- RDKit ----------------
mol = None
if RDKIT_AVAILABLE:
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Molecular Properties")
    st.dataframe(df.T)

with col2:
    st.subheader("Structure")
    if RDKIT_AVAILABLE and mol:
        try:
            st.image(Draw.MolToImage(mol))
        except:
            st.warning("Rendering failed")
    else:
        st.warning("RDKit not available")

# ---------------- MODEL ----------------
scaled = scaler.transform(df) * 2.5
vec = np.concatenate((scaled.flatten(), fp))
X = pd.DataFrame([vec], columns=ALL_FEATURE_NAMES)

prob = model.predict_proba(X)[0][1]

# ---------------- RULE LOGIC ----------------
aromatic = desc.get("NumAromaticRings", 0)
donors = desc.get("NumHDonors", 0)
acceptors = desc.get("NumHAcceptors", 0)
logp = desc.get("LogP", 0)

score = 0

if aromatic >= 1:
    score += 0.25
if aromatic >= 2:
    score += 0.25
if aromatic >= 3:
    score += 0.2

if donors + acceptors > 2:
    score -= 0.1

adj = (prob ** 0.6) + score * (1 - prob)
adj = max(0, min(adj, 0.98))

# ---------------- OUTPUT ----------------
st.subheader("Prediction")

if adj < 0.3:
    st.success("Low AR Activity")
elif adj < 0.6:
    st.warning("Moderate AR Activity")
else:
    st.error("High AR Activity")

st.metric("Probability", f"{adj*100:.1f}%")

# ---------------- INTERPRETATION ----------------
st.subheader("Structural Insights")

insights = []

if logp > 3:
    insights.append("High lipophilicity")
if aromatic > 0:
    insights.append("Aromatic structure present")
if donors > 0 or acceptors > 0:
    insights.append("Hydrogen bonding capability")

if insights:
    for i in insights:
        st.write("✔", i)
else:
    st.write("No major signals detected")

# ---------------- BIO INTERPRETATION ----------------
st.subheader("Biological Meaning")

if adj > 0.7:
    st.error("Likely endocrine disruptor")
elif adj > 0.3:
    st.warning("Moderate interaction potential")
else:
    st.success("Low interaction risk")

# ---------------- SHAP ----------------
st.subheader("Explainability")

try:
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1][0]
    else:
        shap_vals = shap_vals[0]

    idx = np.argsort(np.abs(shap_vals))[-5:]

    for i in idx:
        st.write(f"{ALL_FEATURE_NAMES[i]} → {shap_vals[i]:.3f}")

except:
    st.warning("SHAP not available")

# ---------------- FOOTER ----------------
st.info("Fallback mode ensures stability if RDKit unavailable")
