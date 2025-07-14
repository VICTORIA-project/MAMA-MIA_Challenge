import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import radiomics
import logging

# Suppress non-critical logs
radiomics.logger.setLevel(logging.ERROR)

from xgboost import XGBClassifier, plot_importance, XGBRFClassifier
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, roc_curve
from scipy.ndimage import binary_erosion, label
from scipy.spatial import ConvexHull
from sklearn.preprocessing import LabelEncoder

from radiomics import featureextractor

import pandas as pd
import shap
##################################################

# Paths
features_csv = "/home/hadeel/MAMA-MIA_Challenge/machine_learning/5_folds/all_features.csv"  # Replace with your full path
folds_json = "/home/hadeel/MAMA-MIA_Challenge/machine_learning/5_folds/5fold_split_stratified_updated.json"
output_model_dir = "/home/hadeel/MAMA-MIA_Challenge/machine_learning/5_folds/"

# Load features
features_df = pd.read_csv(features_csv)

# Encode label
le = LabelEncoder()
features_df["pcr"] = le.fit_transform(features_df["pcr"])  # assumes 0/1 or 'No pCR'/'pCR'

# Load folds
with open(folds_json, "r") as f:
    folds = json.load(f)

# Train and save model for each fold
for fold_name, fold_data in folds.items():
    train_ids = [os.path.basename(entry["mask"]).split(".")[0] for entry in fold_data["train"]]
    val_ids = [os.path.basename(entry["mask"]).split(".")[0] for entry in fold_data["val"]]

    # Select rows from features_df
    train_df = features_df[features_df["patient_id"].isin(train_ids)].copy()
    val_df = features_df[features_df["patient_id"].isin(val_ids)].copy()

    X_train = train_df.drop(columns=["patient_id", "pcr"])
    y_train = train_df["pcr"]
    X_val = val_df.drop(columns=["patient_id", "pcr"])
    y_val = val_df["pcr"]

    # === Train XGBoost ===
    ratio = sum(y_train == 0) / sum(y_train == 1)
    model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=1,
        n_estimators=500,
        max_depth=5,
        eval_metric='auc', 
        max_delta_step=1,
        tree_method='hist', enable_categorical=False
    )
    
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(output_model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_model_dir, f"{fold_name}/xgb_model.pkl"))

    # Optional: Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
   
    # === Save predictions ===
    df_preds = pd.DataFrame({
        "patient_id": val_ids,
        "true_label": y_val,
        "pred_label": y_pred,
        "prob_class_1": y_prob
    })
    
    df_preds.to_csv(f"/home/hadeel/MAMA-MIA_Challenge/machine_learning/5_folds/validation_predictions_{fold_name}.csv", index=False)
    print("📁 Predictions saved to validation_predictions.csv")