# 🧬 AI Bioactivity Predictor (NR-AR)

Predict whether a molecule activates the Androgen Receptor (AR) using Machine Learning.

🚀 Early-stage drug screening tool to identify bioactive compounds before expensive lab testing.

---

## 🎯 Demo Output

| Input (SMILES) | Prediction | Confidence |
|---------------|-----------|------------|
| CCO | Low AR Activity | 0.7% |
| c1ccccc1O | Low AR Activity | 4.1% |
| Complex steroid-like compound | High AR Activity | 98% |

---

## 📸 App Preview
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/de257634-f814-4444-b5b2-a75f172e23fc" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3ad99496-8d75-43cf-9256-76f32d675273" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e8a429fd-4225-4106-805f-89160532597d" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/dc937032-93b0-4ff6-8599-9a3f2ba98943" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/32f48660-64c3-4e5c-9dd0-ca49f9357ab0" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e284cf26-897c-4680-9eee-b13df955bfc0" />



---

## 🧠 Overview

This project predicts whether a chemical compound shows activity against the Androgen Receptor (AR) using machine learning.

It serves as an early-stage screening tool to identify potentially bioactive molecules before costly experimental validation.

---

## 🚀 Features

- 🔬 Predicts androgen receptor (AR) activity
- 🧪 Accepts SMILES string input
- 📊 Uses 2000+ molecular features
- 🧠 Ensemble Machine Learning model
- 📉 Probability + confidence score
- 🧬 Molecular structure visualization
- 🧭 Chemical space validation (Tanimoto similarity)
- 📈 Explainability using SHAP

---

## ⚙️ How It Works

1. User inputs SMILES string
2. RDKit extracts molecular descriptors
3. Morgan fingerprints (2048 bits) generated
4. Features passed into trained ML ensemble
5. Output:
   - AR Activity (Low / High)
   - Probability score
   - Chemical similarity validation

---

## 🏆 Why This Matters

- Reduces drug discovery cost
- Enables early detection of bioactive compounds
- Minimizes late-stage failures
- Supports AI-driven pharmaceutical research

---

## ⚙️ Tech Stack

- Python
- Streamlit
- RDKit
- Scikit-learn
- SHAP

---
