import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, roc_curve

# Load JSON
with open("/home/hadeel/MAMA-MIA_Challenge/train_val_split_fold0_reformatted.json", "r") as f:
    data = json.load(f)

def extract_features(img_path, mask_path):
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)

    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)

    roi = img_array[mask_array > 0]
    if roi.size == 0:
        return None  # skip empty masks

    features = {
        "mean": roi.mean(),
        "std": roi.std(),
        "min": roi.min(),
        "max": roi.max(),
        "percentile_25": np.percentile(roi, 25),
        "percentile_75": np.percentile(roi, 75),
        "volume_voxels": np.sum(mask_array > 0),
    }
    return list(features.values())

def load_data(entries):
    X, y, ids = [], [], []
    for entry in tqdm(entries):
        img_path = entry["image"][0]
        mask_path = entry["mask"]
        label = entry["pcr"]
        pid = os.path.basename(mask_path).split(".")[0]

        feats = extract_features(img_path, mask_path)
        if feats is not None:
            X.append(feats)
            y.append(label)
            ids.append(pid)
    return np.array(X), np.array(y), ids

# === Load JSON ===
with open("train_val_split_fold0_reformatted.json", "r") as f:
    data = json.load(f)

train_entries = data["fold_0"]["train"]
val_entries = data["fold_0"]["val"]

# === Extract features ===
print("Extracting training features...")
X_train, y_train, ids_train = load_data(train_entries)

print("Extracting validation features...")
X_val, y_val, ids_val = load_data(val_entries)

# === Normalize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# === Train XGBoost ===
ratio = sum(y_train == 0) / sum(y_train == 1)
model = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=ratio,
    n_estimators=100,
    max_depth=4,
    use_label_encoder=False,
    eval_metric="logloss",
    max_delta_step=1
)
model.fit(X_train_scaled, y_train)

# === Save model and scaler ===
joblib.dump(model, "/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/xgb_pcr_model.pkl")
joblib.dump(scaler, "/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/scaler.pkl")
print("✅ Model and scaler saved.")

# === Predict on validation ===
y_pred = model.predict(X_val_scaled)
y_prob = model.predict_proba(X_val_scaled)[:, 1]

# Basic metrics
balanced_acc = balanced_accuracy_score(y_val, y_pred)
auc_score = roc_auc_score(y_val, y_prob)
print("\n📊 Validation Performance:")
print(classification_report(y_val, y_pred))
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"AUC: {auc_score:.4f}")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_prob)

# Specificity @ 90% sensitivity
target_sensitivity = 0.90
idx_sens = np.argmin(np.abs(tpr - target_sensitivity))
spec_at_90sens = 1 - fpr[idx_sens]
thresh_at_90sens = thresholds[idx_sens]

# Sensitivity @ 90% specificity
target_specificity = 0.90
specificities = 1 - fpr
idx_spec = np.argmin(np.abs(specificities - target_specificity))
sens_at_90spec = tpr[idx_spec]
thresh_at_90spec = thresholds[idx_spec]

print(f"\n🔬 Specificity at 90% Sensitivity: {spec_at_90sens:.4f} (threshold={thresh_at_90sens:.4f})")
print(f"🔬 Sensitivity at 90% Specificity: {sens_at_90spec:.4f} (threshold={thresh_at_90spec:.4f})")


plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Validation Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/roc_curve_validation.png")
plt.show()
print("📈 ROC curve saved as roc_curve_validation.png")

# === Save predictions ===
import pandas as pd
df_preds = pd.DataFrame({
    "patient_id": ids_val,
    "true_label": y_val,
    "pred_label": y_pred,
    "prob_class_1": y_prob
})
df_preds.to_csv("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/validation_predictions.csv", index=False)
print("📁 Predictions saved to validation_predictions.csv")