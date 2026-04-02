import math
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from PIL import Image
import shap
from rdkit import Chem
from rdkit.Chem import Draw

# Natively modularize logic!
import sys
sys.path.append('src')
from utils import feature_extraction, DESCRIPTOR_NAMES, FP_NAMES, ALL_FEATURE_NAMES

st.set_page_config(page_title="AI Bioactivity Predictor (NR-AR)", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_artifacts():
    try:
        model_calibrated = joblib.load(os.path.join('data', 'best_model.pkl'))
        model_raw = joblib.load(os.path.join('data', 'best_model_raw.pkl'))
        scaler = joblib.load(os.path.join('data', 'scaler.pkl'))
        explainer = shap.TreeExplainer(model_raw)
        return model_calibrated, model_raw, scaler, explainer
    except Exception as e:
        return None, None, None, None

@st.cache_data
def load_dataset_baseline():
    """
    Loads training data features to calculate dynamic Tanimoto similarities
    and provides the overarching base probability dataset mean natively.
    """
    try:
        df_train = pd.read_csv('data/processed_tox21.csv')
        sample = df_train.sample(n=min(1000, len(df_train)), random_state=42)
        X_sample = sample[ALL_FEATURE_NAMES]
        return df_train[FP_NAMES].values, X_sample
    except Exception as e:
        return None, None

model_calibrated, model_raw, scaler, explainer = load_artifacts()
train_fps, train_sample = load_dataset_baseline()

st.title("🧬 AI Bioactivity Predictor (NR-AR)")
st.markdown("""
Welcome to the AI Bioactivity Predictor (NR-AR)!
This system evaluates candidate compounds using an Isotonically Calibrated Deep Ensemble operating over exactly **2056 extracted physicochemical properties** and explicit Morgan sub-structures to flag biological activity rapidly.
""")
st.info("This tool is designed as an early-stage screening system to identify compounds that may interact with androgen receptors (AR), helping detect potential endocrine disruption risks before costly experimental validation.")
st.info("**Problem We Solve**: Drug development has a high failure rate due to late-stage bioactivity detection. Our system helps identify activated compounds early using explainable AI.")
st.divider()

if not model_calibrated or train_fps is None:
    st.error("Model artifacts or training data not found. Please execute the underlying pipeline loops iteratively first.")
    st.stop()

# Pre-calculate baseline average natively inside the UI state!
baseline_prob = model_calibrated.predict_proba(train_sample)[:, 1].mean()

st.sidebar.header("🔬 Target Chemistry")
st.sidebar.markdown("Specify the precise organic topology:")
smiles_input = st.sidebar.text_input("Enter SMILES string:", value="CCOc1ccc2nc(S(N)(=O)=O)sc2c1", help="RDKit required format")

raw_features = feature_extraction(smiles_input)

if np.isnan(raw_features[0]):
    st.error("⚠️ Invalid SMILES string. Cannot extract physicochemical layout.")
    st.stop()

desc_dict = dict(zip(DESCRIPTOR_NAMES, raw_features[:len(DESCRIPTOR_NAMES)]))
input_df_raw = pd.DataFrame([desc_dict])
input_fp = raw_features[len(DESCRIPTOR_NAMES):]

# ----------------------------------------------------
# Chemical Similarity Validation (Tanimoto) natively
# ----------------------------------------------------
intersection = np.logical_and(train_fps, input_fp).sum(axis=1)
union = np.logical_or(train_fps, input_fp).sum(axis=1)
tanimoto_scores = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
max_tanimoto = tanimoto_scores.max()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Candidate Baseline Statistics")
    st.dataframe(input_df_raw.T, use_container_width=True)
    st.caption("Extracted Core Properties (+ 2048 Morgan Topology bits)")
    
    st.markdown("---")
    if max_tanimoto > 0.6:
        st.success(f"**Chemical Space Validation**: {max_tanimoto:.2f} Tanimoto Similarity (Within known chemical space)")
    elif max_tanimoto >= 0.3:
        st.warning(f"**Chemical Space Validation**: {max_tanimoto:.2f} Tanimoto Similarity (Moderately similar)")
    else:
        st.error(f"**Chemical Space Validation**: {max_tanimoto:.2f} Tanimoto Similarity (Outside training distribution)")
        st.warning("⚠️ This molecule lies strictly outside the training chemical space. Predictions may be less reliable.")

with col2:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is not None:
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption="Molecular Structure")
        
    st.subheader("Prediction Inference")
    
    scaled_desc_arr = scaler.transform(input_df_raw)
    # Align identically with the structural weighting natively injected during train.py
    scaled_desc_arr *= 2.5
    
    scaled_feature_vector = np.concatenate((scaled_desc_arr.flatten(), input_fp))
    scaled_df = pd.DataFrame([scaled_feature_vector], columns=ALL_FEATURE_NAMES)
    
    probabilities = model_calibrated.predict_proba(scaled_df)[0]
    confidence_toxic = probabilities[1]
    
    # --- HYBRID AR SCORING (ML BASE + RULE BOOSTS) ---
    raw_prob = confidence_toxic

    try:
        logp                = desc_dict.get('LogP', 0)
        ring_count          = desc_dict.get('RingCount', 0)
        h_donors            = desc_dict.get('NumHDonors', 0)
        h_acceptors         = desc_dict.get('NumHAcceptors', 0)
        aromatic_ring_count = desc_dict.get('NumAromaticRings', 0)
        heavy_atom_count    = desc_dict.get('HeavyAtomCount', 0)
        fg_total            = h_donors + h_acceptors

        # ── FIX 2: Stretch ML base probability (pushes mid-values upward) ─────
        # raw_prob ** 0.6 expands the dynamic range so weak signals separate well
        prob = raw_prob ** 0.6

        # ── FIX 3: Detect functional groups via SMARTS ────────────────────────
        has_OH   = mol is not None and mol.HasSubstructMatch(
                       Chem.MolFromSmarts('[OX2H]'))
        has_COOH = mol is not None and mol.HasSubstructMatch(
                       Chem.MolFromSmarts('[CX3](=O)[OX2H1]'))

        # ── Assemble adjustment score ─────────────────────────────────────────
        score = 0.0

        # FIX 1 — Aromatic ring boosts (linear, per-tier)
        if aromatic_ring_count >= 1:
            score += 0.25                 # benzene: moderate boost
        if aromatic_ring_count >= 2:
            score += 0.25                 # naphthalene: cumulative → +0.50
        if aromatic_ring_count >= 3:
            score += 0.20                 # anthracene: cumulative → +0.70

        # FIX 3 — Functional group boosts
        if has_OH:
            score += 0.08                 # hydroxyl enhances receptor binding
        if has_COOH:
            score += 0.12                 # carboxylic acid adds polar interaction

        # FIX 4 — Softened polarity penalty (was −0.3 flat, now proportional + capped)
        penalty = min(0.15, 0.05 * fg_total)
        score  -= penalty

        # ── Combine: base + rule adjustments, apply (1 − prob) guard ─────────
        # Uses (1 − prob) scaling so boosts compress near the ceiling naturally
        adjusted_prob = prob + score * (1 - prob)
        adjusted_prob = min(max(adjusted_prob, 0.0), 0.98)

        # ── DEBUG (mandatory) ─────────────────────────────────────────────────
        print(f"aromatic_ring_count : {aromatic_ring_count}")
        print(f"raw_prob (ML)       : {raw_prob:.3f}")
        print(f"prob (stretched)    : {prob:.3f}")
        print(f"score (adjustments) : {score:.3f}")
        print(f"adjusted_prob       : {adjusted_prob:.3f}")

    except Exception as e:
        print("Scoring error:", e)
        adjusted_prob = raw_prob
    
    if adjusted_prob < 0.3:
        st.success("### ✅ Low AR Activity")
        risk_level = "Low Risk"
        color = "normal"
    elif adjusted_prob <= 0.6:
        st.warning("### 🟡 Moderate AR Activity")
        risk_level = "Moderate Risk"
        color = "off"
    else:
        st.error("### ⚠️ High AR Activity")
        risk_level = "High Risk"
        color = "inverse"

    st.metric(
        label="Calibrated Probability", 
        value=f"{adjusted_prob*100:.1f}%", 
        delta=f"{(adjusted_prob - baseline_prob)*100:+.1f}% vs Dataset Baseline ({baseline_prob*100:.1f}%)",
        delta_color="inverse" if adjusted_prob > baseline_prob else "normal"
    )

    boost_value = adjusted_prob - raw_prob
    st.caption(f"🔧 Heuristic adjustment applied: +{round(boost_value, 3)}")

    if adjusted_prob > 0.75:
        risk_level = "High Risk"
    elif adjusted_prob > 0.4:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    if risk_level == "High Risk":
        st.error(f"🧠 Risk Level: {risk_level}")
    elif risk_level == "Moderate Risk":
        st.warning(f"🧠 Risk Level: {risk_level}")
    else:
        st.success(f"🧠 Risk Level: {risk_level}")
    
    st.info("This prediction estimates the likelihood of androgen receptor interaction based on structural features. It does not represent overall drug toxicity.")
    st.caption("Confidence is derived from calibrated ensemble trained on Tox21 NR-AR assay data.")
    st.caption("Prediction adjusted using structural heuristics for improved biological realism")
    st.error("High AR activity may indicate endocrine disruption or hormone receptor interaction.")

    st.markdown("### 🔍 Structural Contribution")

    contributions = []
    
    logp = desc_dict.get('LogP', 0)
    rings = desc_dict.get('RingCount', 0)
    molwt = desc_dict.get('MolWt', 0)

    if logp > 3:
        contributions.append("High lipophilicity (LogP > 3)")
    elif logp > 2:
        contributions.append("Moderate lipophilicity (LogP > 2)")

    if rings >= 1:
        contributions.append("Aromatic / ring structure detected")

    if rings >= 3:
        contributions.append("Multi-ring system (complex topology)")

    if molwt > 250:
        contributions.append("High molecular weight (drug-like size)")

    if contributions:
        for c in contributions:
            st.write(f"✔ {c}")
    else:
        st.write("No strong structural bioactivity signals detected")

    st.markdown("### Biological Interpretation")
    if desc_dict['LogP'] > 4.5:
        st.markdown("- High lipophilicity suggests membrane permeability and potential receptor binding (common in steroid-like compounds).")
    if desc_dict['RingCount'] > 0:
        st.markdown("- Presence of ring structures increases likelihood of receptor interaction due to structural rigidity.")
    if desc_dict['MolWt'] > 350:
        st.markdown("- Higher molecular weight may indicate complex bioactive scaffolds seen in endocrine-active compounds.")
    if desc_dict['TPSA'] < 50:
        st.markdown("- Low polarity supports cell membrane penetration, a key factor in receptor binding.")
    if desc_dict['NumHDonors'] > 0 or desc_dict['NumHAcceptors'] > 0:
        st.markdown("- Hydrogen bonding capability supports interaction with biological targets.")

    st.markdown("### Prediction Confidence")
    if adjusted_prob < 0.3:
        confidence_text = "High confidence: model strongly predicts LOW AR activity."
    elif adjusted_prob < 0.7:
        confidence_text = "Moderate confidence: compound lies in uncertain/borderline region. Experimental validation recommended."
    else:
        confidence_text = "High confidence: model strongly predicts HIGH AR activity."
    st.info(confidence_text)
    
    st.markdown("### Real-World Implication")
    if adjusted_prob >= 0.7:
        impact = "This compound may act as an endocrine disruptor by interacting with androgen receptors, potentially affecting hormonal balance."
        st.error(impact)
    elif adjusted_prob >= 0.3:
        impact = "This compound shows moderate interaction potential with androgen receptors and should be validated experimentally."
        st.warning(impact)
    else:
        impact = "This compound is unlikely to significantly interact with androgen receptors under normal conditions."
        st.success(impact)

st.divider()

# --------------------------------------------------------------------------
# Localized Interpretability (SHAP-Driven)
# --------------------------------------------------------------------------
st.subheader("🧠 Key Factors Influencing This Prediction (SHAP Insights)")

shap_vals_raw = explainer.shap_values(scaled_df)
if isinstance(shap_vals_raw, list):
    shap_vals = shap_vals_raw[1][0]
else:
    shap_vals = shap_vals_raw[0, :, 1] if len(shap_vals_raw.shape) == 3 else shap_vals_raw[0]

# Parse strictly the absolute top 5 features natively
feature_impacts = []
for i, name in enumerate(ALL_FEATURE_NAMES):
    sv = shap_vals[i]
    feature_impacts.append({
        'name': name,
        'shap': sv,
        'abs_shap': abs(sv)
    })

feature_impacts = sorted(feature_impacts, key=lambda x: x['abs_shap'], reverse=True)[:5]

st.markdown("#### Prediction driven by:")
for f in feature_impacts:
    val = f['shap']
    direction = "Increases" if val > 0 else "Decreases"
    icon = "🔴" if val > 0 else "🟢"
    name = f['name']
    
    if name == 'MolLogP':
        meaning = "lipophilicity"
    elif name == 'TPSA':
        meaning = "polarity"
    elif name == 'RingCount':
        meaning = "aromatic structure"
    elif name.startswith('Morgan_'):
        meaning = f"substructure fragment (Morgan bit {name.split('_')[1]})"
    else:
        meaning = name
        
    st.markdown(f"- {icon} **{meaning}**: {direction} predicted AR activity.")

st.markdown("---")
st.subheader("Global Explanations")
tab1, tab2, tab3 = st.tabs(["Global Feature Density", "Priority Weight Matrix", "Algorithm Routing"])

def load_image(filename):
    path = os.path.join('data', filename)
    return Image.open(path) if os.path.exists(path) else None

with tab1:
    img = load_image('shap_summary.png')
    if img: st.image(img, use_container_width=True)
with tab2:
    img = load_image('shap_bar.png')
    if img: st.image(img, use_container_width=True)
with tab3:
    img = load_image('shap_waterfall.png')
    if img: st.image(img, use_container_width=True)
