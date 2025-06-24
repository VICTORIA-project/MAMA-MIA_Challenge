import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score, balanced_accuracy_score, roc_curve
)
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

# === Reload Feature Extractor ===
# --- 1. Feature extraction returns a dict (no NaNs) ---
def extract_features(img_path, mask_path):
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)

    # If the predicted mask is empty → segmentation_failed
    if np.sum(mask_array > 0) == 0:
        return {
            "volume_vox":       0,
            "surface_vox":      0,
            "bbox_volume":      0,
            "compactness":      0,   # set 0 instead of NaN
            "extent":           0,   # set 0 instead of NaN
            "segmentation_failed": 1
        }

    # Otherwise compute your shape features
    volume_vox = np.sum(mask_array > 0)
    eroded     = binary_erosion(mask_array)
    surface_vox = np.sum(mask_array != eroded)

    coords     = np.argwhere(mask_array)
    bbox_min   = coords.min(axis=0)
    bbox_max   = coords.max(axis=0)
    bbox_dims  = bbox_max - bbox_min + 1
    bbox_volume = np.prod(bbox_dims)

    compactness = (volume_vox**2) / max(1, surface_vox**3)
    extent      = volume_vox / max(1, bbox_volume)

    return {
    "volume_vox": volume_vox,
    "surface_vox": surface_vox,
    "bbox_volume": bbox_volume,
    "compactness": compactness,
    "extent": extent,
    "segmentation_failed": 0
    }
    
feature_names = [
    "volume_vox",
    "surface_vox",
    "bbox_volume",
    "compactness",
    "extent",
    "segmentation_failed"
]
numeric_cols = [
    "volume_vox",
    "surface_vox",
    "bbox_volume",
    "compactness",
    "extent"
]
    
def load_data(entries):
    X, y, ids = [], [], []
    for entry in tqdm(entries):
        img_path = entry["image"][0]
        mask_path = entry["mask"]
        label = entry["pcr"]
        pid = os.path.basename(mask_path).split(".")[0]

        feats = extract_features(img_path, mask_path)
        if feats is not None:
            X.append([feats[f] for f in feature_names])
            y.append(label)
            ids.append(pid)
    return pd.DataFrame(X, columns=feature_names), np.array(y), ids

# === Load model and scaler ===
model = joblib.load("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/xgb_pcr_model.pkl")
scaler = joblib.load("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/scaler.pkl")

# === Load validation set ===
with open("train_val_split_fold0_reformatted.json", "r") as f:
    data = json.load(f)
val_entries = data["fold_0"]["val"]

print("Extracting validation features...")
val_df, y_val, ids_val = load_data(val_entries)

X_val_scaled = scaler.transform(val_df[numeric_cols])
X_val_full = np.hstack([X_val_scaled, val_df[["segmentation_failed"]].values])

X_val_df = pd.DataFrame(X_val_full, columns=feature_names)


# === Predict on validation ===
# Predict
y_pred_all = model.predict(X_val_df)
y_prob_all = model.predict_proba(X_val_df)[:, 1]

# Fallback rule for failed cases
y_pred = y_pred_all.copy()
y_prob = y_prob_all.copy()
failed_mask = X_val_df["segmentation_failed"] == 1
y_pred[failed_mask] = 0
y_prob[failed_mask] = 0.5

# === Evaluation
print("\n📊 Reproduced Evaluation:")
print(classification_report(y_val, y_pred))
print(f"Balanced Accuracy: {balanced_accuracy_score(y_val, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_val, y_prob):.4f}")

# Specificity @ 90% Sensitivity
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
target_sensitivity = 0.90
idx_sens = np.argmin(np.abs(tpr - target_sensitivity))
spec_at_90sens = 1 - fpr[idx_sens]

# Sensitivity @ 90% Specificity
specificities = 1 - fpr
target_specificity = 0.90
idx_spec = np.argmin(np.abs(specificities - target_specificity))
sens_at_90spec = tpr[idx_spec]

print(f"🔬 Specificity at 90% Sensitivity: {spec_at_90sens:.4f}")
print(f"🔬 Sensitivity at 90% Specificity: {sens_at_90spec:.4f}")

# === ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_val, y_prob):.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.scatter(fpr[idx_sens], tpr[idx_sens], color='red', label="90% Sensitivity")
plt.scatter(fpr[idx_spec], tpr[idx_spec], color='blue', label="90% Specificity")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Validation Set")
plt.legend()
plt.tight_layout()
# plt.savefig("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/exp-1/roc_curve_reproduced.png")
plt.show()

# === Save predictions
df = pd.DataFrame({
    "patient_id": ids_val,
    "true_label": y_val,
    "pred_label": y_pred,
    "prob_class_1": y_prob
})
# df.to_csv("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/exp-1/validation_predictions_reproduced.csv", index=False)
# print("✅ Saved predictions to validation_predictions_reproduced.csv")
