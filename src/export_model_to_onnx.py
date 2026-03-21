#!/usr/bin/env python3
"""
Export Gradient Boosting model to ONNX for browser-based calculator.

This script demonstrates the full pipeline:
1. Load and prepare training data
2. Engineer features
3. Train a Gradient Boosting Classifier with preprocessing
4. Export the trained model to ONNX format
5. Save preprocessing parameters as JSON for the web calculator

Note: Patient-level data is not included in this repository.
Replace 'data/training_data.csv' with your own institutional data
formatted as described in data/README.md.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import json
import warnings

warnings.filterwarnings("ignore")


def load_data(data_path):
    """Load and validate the training dataset."""
    df = pd.read_csv(data_path)
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    return df


def engineer_features(X):
    """Create derived features from base clinical variables."""
    if "age" in X.columns and "plt_count" in X.columns:
        X["age_plt_interaction"] = X["age"] * X["plt_count"] / 10000

    if (
        "sdh_thickness__receiving_mmae" in X.columns
        and "midline_shift__if_no_put_0" in X.columns
    ):
        X["sdh_shift_product"] = (
            X["sdh_thickness__receiving_mmae"] * X["midline_shift__if_no_put_0"] / 100
        )
        X["sdh_shift_ratio"] = X["midline_shift__if_no_put_0"] / (
            X["sdh_thickness__receiving_mmae"] + 1
        )

    if "plt_count" in X.columns and "hemoglobin__at_admission" in X.columns:
        X["plt_hgb_ratio"] = X["plt_count"] / (X["hemoglobin__at_admission"] + 0.1)

    if "age" in X.columns:
        X["elderly"] = (X["age"] > 80).astype(int)
        X["age_squared"] = (X["age"] / 100) ** 2
        if "charlson_score" in X.columns:
            X["age_charlson_product"] = X["age"] * X["charlson_score"] / 100

    # Composite bleeding risk
    risk_components = []
    if "plt_count" in X.columns:
        risk_components.append((X["plt_count"] < 100).astype(int) * 2)
        risk_components.append((X["plt_count"] < 150).astype(int))
    if "anticoagulants" in X.columns:
        risk_components.append(X["anticoagulants"].fillna(0).astype(int))
    if "bleeding_risk_score" in X.columns:
        risk_components.append((X["bleeding_risk_score"] > 2).astype(int))
    if risk_components:
        X["composite_bleeding_risk"] = sum(risk_components)

    # Neurological severity
    neuro_components = []
    if "admission_gcs" in X.columns:
        neuro_components.append((X["admission_gcs"] < 15).astype(int))
        neuro_components.append((X["admission_gcs"] < 13).astype(int) * 2)
    if "significant_mass_effect" in X.columns:
        neuro_components.append(
            X["significant_mass_effect"].fillna(0).astype(int) * 2
        )
    if "altered_mental_status" in X.columns:
        neuro_components.append(X["altered_mental_status"].fillna(0).astype(int))
    if neuro_components:
        X["neurological_severity"] = sum(neuro_components)

    return X


def main():
    # --- Configuration ---
    DATA_PATH = "data/training_data.csv"
    OUTPUT_ONNX = "calculator/model.onnx"
    OUTPUT_PARAMS = "calculator/model_params.json"

    # Pre-treatment base features
    pre_treatment_features = [
        "age", "sex", "ethnicity",
        "admission_gcs", "admission_premorbid__mrs",
        "altered_mental_status", "confusion", "headache", "seizure",
        "focal_weakness", "facial_droop", "cranial_nerve_palsy",
        "plt_count", "hemoglobin__at_admission", "wbc__at_admission",
        "aptt__at_admission",
        "sdh_thickness__receiving_mmae", "midline_shift__if_no_put_0",
        "sdh_laterality", "brain_atrophy", "membranes_or_septations",
        "sdh_imaging_characteristics", "cause_of_sdh",
        "cardiac_disease", "liver_disease", "history_of_gastrointestinal_bleeding",
        "history_of_vte", "prior_diagnosis_of_icad", "hx_of_cranial_trauma",
        "hx_ventriculoperitoneal_shunt", "sickle_cell_disease", "hivhcv",
        "anticoagulants", "antiplatelets", "antithrombotics",
        "anticoagulation_drug",
        "alcohol", "smoking",
        "charlson_score", "charlson_age_adjusted",
        "anticoag_risk_score", "bleeding_risk_score",
        "is_chronic_sdh", "is_acute_sdh", "is_mixed_sdh", "is_bilateral",
        "complex_sdh", "significant_mass_effect",
        "severe_thrombocytopenia", "moderate_thrombocytopenia",
        "severe_anemia", "moderate_anemia", "prolonged_aptt",
        "coagulopathy_score", "thrombocytopenia",
    ]

    # --- Load data ---
    df = load_data(DATA_PATH)
    available_features = [f for f in pre_treatment_features if f in df.columns]
    print(f"Available features: {len(available_features)}")

    X = df[available_features].copy()
    y = df["treatment_failure"].copy()

    # --- Feature engineering ---
    X = engineer_features(X)
    feature_names = list(X.columns)
    print(f"Total features after engineering: {X.shape[1]}")

    # --- Handle missing values ---
    for col in X.columns:
        if X[col].dtype in ["int64", "float64"]:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(0)

    # --- Preprocessing ---
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    scaler.fit(X_imputed)

    imputer_medians = imputer.statistics_.tolist()

    # --- Train/test split ---
    stratify_var = (
        y.astype(str)
        + "_"
        + pd.qcut(X["age"], q=3, labels=["0", "1", "2"], duplicates="drop").astype(
            str
        )
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=stratify_var
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    X_train_scaled = scaler.transform(imputer.transform(X_train))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    # --- Train model ---
    gb_model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
    )
    gb_model.fit(X_train_scaled, y_train)

    y_pred_uncal = gb_model.predict_proba(X_test_scaled)[:, 1]
    auc_uncal = roc_auc_score(y_test, y_pred_uncal)
    print(f"Uncalibrated AUC: {auc_uncal:.3f}")

    # --- Calibrate ---
    calibrated_model = CalibratedClassifierCV(gb_model, method="isotonic", cv=3)
    calibrated_model.fit(X_train_scaled, y_train)

    y_pred_cal = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    auc_cal = roc_auc_score(y_test, y_pred_cal)
    brier = brier_score_loss(y_test, y_pred_cal)
    print(f"Calibrated AUC: {auc_cal:.3f}")
    print(f"Brier Score: {brier:.3f}")

    # --- Export to ONNX ---
    initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(gb_model, initial_types=initial_type, target_opset=12)

    with open(OUTPUT_ONNX, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to: {OUTPUT_ONNX}")

    # --- Verify ONNX model ---
    sess = ort.InferenceSession(OUTPUT_ONNX)
    input_name = sess.get_inputs()[0].name
    test_input = X_test_scaled[:5].astype(np.float32)
    onnx_outputs = sess.run(None, {input_name: test_input})
    onnx_pred = np.array([p[1] for p in onnx_outputs[1]])
    sklearn_pred = gb_model.predict_proba(test_input)[:, 1]
    print(f"ONNX vs sklearn match: {np.allclose(sklearn_pred, onnx_pred, atol=1e-5)}")

    # --- Save parameters for web calculator ---
    params = {
        "feature_names": feature_names,
        "imputer_medians": imputer_medians,
        "auc": float(auc_uncal),
        "note": "ONNX model includes preprocessing (imputer + scaler). Pass RAW values to model.",
    }

    with open(OUTPUT_PARAMS, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Model parameters saved to: {OUTPUT_PARAMS}")


if __name__ == "__main__":
    main()
