# Hospital Readmission Risk Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-orange?logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-interpretability-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Complete-success)

A machine learning project to predict 30-day hospital readmission risk for diabetic patients, using the UCI Diabetes 130-US Hospitals dataset. The goal is to rank patients by their predicted readmission risk at the point of discharge, enabling clinical teams to prioritise follow-up interventions for the highest-risk patients.

📋 Full project report — methodology, analysis, results, and commentary are documented in [Final Report](./notebooks/03_final_report.ipynb). This is the recommended starting point for reviewing the project end-to-end

---

## Project Structure

```
hospital-readmission-risk/
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
│   ├── models/                           # Model artefacts saved from notebook runs
│   ├── 01_data_cleaning_prep.ipynb       # Data cleaning, EDA & feature engineering
│   ├── 02_model_training_and_evaluation.ipynb  # Modelling, evaluation & analysis
│   └── 03_final_report.ipynb             # Final report notebook
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── tables/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Problem Statement

30-day readmission is a key quality-of-care metric and a significant cost driver for hospitals. This project frames the problem as **binary classification**: predicting whether a diabetic patient will be readmitted within 30 days of discharge. With only 11.4% of encounters resulting in a 30-day readmission, the dataset is inherently imbalanced, which required deliberate handling at both training time and threshold selection.

---

## Key Results

- **Final model:** XGBoost (Reduced, 69 features) with `scale_pos_weight` for class imbalance
- **Test AUC: 0.6673** on the held-out test set
- **Top decile lift: 2.40x** — the highest-risk 10% of patients have a readmission rate of 26.8% vs. 11.2% population average
- **Top 30% of patients by risk score captures 50.5% of all readmissions** — half of all cases identified by targeting one third of the population
- SHAP analysis confirmed clinically coherent predictions: prior utilisation history (28%), discharge disposition (20%), and encounter intensity (16%) account for 64% of model predictive power

---

## Methodology Summary

### Data Cleaning
- Removed encounters with hospice/death discharge dispositions (2,423 rows)
- Replaced `"?"` sentinel values with `NaN`
- Dropped `weight` (97% missing) and `payer_code` (40% missing)
- Patient-aware train/test split using `GroupShuffleSplit` on `patient_nbr` to prevent data leakage across encounters from the same patient (80/20 split, zero patient overlap)

### Feature Engineering
- Age bucket strings converted to ordinal numeric midpoints
- ICD-9 diagnosis codes mapped to 9 broad clinical categories following Strack et al. (2014)
- Admission type, source, and discharge disposition integer IDs decoded and collapsed into interpretable categories
- Engineered features: `prior_utilization_score`, `any_prior_inpatient/emergency/outpatient`, `institutional_discharge`, `med_intensity`, `lab_intensity`, `n_active_diabetes_meds`, `insulin_flag`, `insulin_change`

### Modelling
Three model families were evaluated, each with class imbalance handled at training time and threshold tuning applied via out-of-fold (OOF) cross-validation predictions:

| Model | Imbalance Strategy | CV AUC |
|---|---|---|
| Unweighted Logistic Regression | None | 0.660 |
| Balanced Logistic Regression | `class_weight="balanced"` | 0.662 |
| LASSO Logistic Regression | `class_weight="balanced"`, L1 C=0.046 | 0.662 |
| XGBoost Baseline | `scale_pos_weight=7.74` | 0.665 |
| XGBoost Tuned (RandomizedSearchCV, 30 iterations) | `scale_pos_weight=7.74` | 0.670 |
| **XGBoost Reduced — Final Model** | **`scale_pos_weight=7.74`** | **0.667** |

The final model was selected after SHAP-guided feature reduction: two clinical feature groups (`admission_context` and `lab_results`) contributing only ~2% of predictive power were dropped, reducing the feature set from 79 to 69 with a test AUC change of just -0.0002.

### Evaluation
- ROC-AUC (primary metric, threshold-independent)
- OOF threshold tuning: recall-oriented threshold selected (highest recall subject to precision ≥ 16%)
- Calibration curve assessment
- Decile-based lift and cumulative gains analysis
- SHAP analysis: beeswarm, grouped clinical importance, and patient-level waterfall plots
- Fairness audit across racial and gender subgroups (AUC, recall, FNR, FPR per group)

---

## How to Run

### 1. Set up the environment

```bash
git clone https://github.com/your-username/hospital-readmission-risk.git
cd hospital-readmission-risk
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the raw data

Download the dataset from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) and place the following files in `data/raw/`:
- `diabetic_data.csv`
- `IDS_mapping.csv`

### 3. Run the notebooks in order

**Notebook 1 — Data Cleaning & Feature Engineering**
```
notebooks/01_data_cleaning_prep.ipynb
```
Performs data cleaning, feature engineering, and patient-aware train/test splitting. Reusable feature engineering logic lives in `src/feature_engineering.py` and is called via the notebook. Outputs processed datasets to `data/processed/`.

**Notebook 2 — Model Training & Evaluation**
```
notebooks/02_model_training_and_evaluation.ipynb
```
Prepares the test set using `src/pipeline.py` and `src/preprocessing.py`, which apply the same feature engineering and one-hot encoding transformations as were applied to the training data, ensuring no leakage:

```python
# Apply feature engineering pipeline to test data
from src.pipeline import run_pipeline
X_test, y_test = run_pipeline(test_df, include_target=True)

# Align test encoding to match training columns
from src.preprocessing import transform_test_to_match_train
X_test_processed = transform_test_to_match_train(
    X_test, reference_columns=X_train.columns.tolist()
)
```

Model training, hyperparameter tuning, threshold optimisation, and all evaluation (SHAP, calibration, fairness, lift analysis) then follow within the notebook. Outputs figures and tables to `outputs/`.

**Notebook 3 — Final Report**
```
notebooks/03_final_report.ipynb
```
Final report consolidating all results, analysis, and commentary.

---

## Dependencies

See `requirements.txt` for the full list. Key libraries:

```
Python 3.12
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
statsmodels
joblib
dataframe_image
```

---

## Dataset

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*.

UCI ML Repository: [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

---

## License

This project is licensed under the [MIT License](LICENSE)

---

## Author

**Bhavana Sanghi**
