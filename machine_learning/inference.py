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
def extract_features(img_path, mask_path):
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)

    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)

    roi = img_array[mask_array > 0]
    if roi.size == 0:
        return None

    # mean = roi.mean()
    # std = roi.std()
    # min_val = roi.min()
    # max_val = roi.max()
    # p25 = np.percentile(roi, 25)
    # p75 = np.percentile(roi, 75)
    volume_vox = np.sum(mask_array > 0)

    eroded = binary_erosion(mask_array)
    surface_vox = np.sum(mask_array != eroded)

    coords = np.argwhere(mask_array)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_dims = bbox_max - bbox_min + 1
    bbox_volume = np.prod(bbox_dims)

    elongation = bbox_dims.max() / max(1, bbox_dims.min())
    compactness = (volume_vox ** 2) / max(1, surface_vox ** 3)
    sphericity = (np.pi ** (1 / 3) * (6 * volume_vox) ** (2 / 3)) / max(1, surface_vox)

    return [
        # mean, std, min_val, max_val, p25, p75, 
        volume_vox,
        surface_vox, 
        bbox_volume,
        elongation, compactness, sphericity
    ]

# === Load model and scaler ===
model = joblib.load("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/exp-4/xgb_pcr_model.pkl")
scaler = joblib.load("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/exp-4/scaler.pkl")

# === Load validation set ===
with open("train_val_split_fold0_reformatted.json", "r") as f:
    data = json.load(f)
val_entries = data["fold_0"]["val"]

# === Feature extraction
X_val, y_val, ids_val = [], [], []

for entry in tqdm(val_entries):
    img_path = entry["image"][0]
    mask_path = entry["mask"]
    label = entry["pcr"]
    pid = os.path.basename(mask_path).split(".")[0]

    feats = extract_features(img_path, mask_path)
    if feats is not None:
        X_val.append(feats)
        y_val.append(label)
        ids_val.append(pid)

X_val = np.array(X_val)
y_val = np.array(y_val)
X_val_scaled = scaler.transform(X_val)

# === Inference
y_pred = model.predict(X_val_scaled)
y_prob = model.predict_proba(X_val_scaled)[:, 1]

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
