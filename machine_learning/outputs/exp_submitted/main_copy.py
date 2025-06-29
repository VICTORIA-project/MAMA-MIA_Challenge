import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from xgboost import XGBClassifier, plot_importance, XGBRFClassifier
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, roc_curve
from scipy.ndimage import binary_erosion, label
from scipy.spatial import ConvexHull

from radiomics import featureextractor

import pandas as pd
import shap

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

# === Load JSON ===
with open("train_val_split_fold0_reformatted.json", "r") as f:
    data = json.load(f)

train_entries = data["fold_0"]["train"]
val_entries = data["fold_0"]["val"]

# Extract features
print("Extracting training features...")
train_df, y_train, ids_train = load_data(train_entries)

print("Extracting validation features...")
val_df, y_val, ids_val = load_data(val_entries)

# === Normalize ===
scaler = StandardScaler()

# Fit only on valid training data
valid_mask_train = train_df["segmentation_failed"] == 0
scaler.fit(train_df.loc[valid_mask_train, numeric_cols])

# Transform all
X_train_scaled = scaler.transform(train_df[numeric_cols])
X_val_scaled = scaler.transform(val_df[numeric_cols])

# Add segmentation_failed back
X_train_full = np.hstack([X_train_scaled, train_df[["segmentation_failed"]].values])
X_val_full = np.hstack([X_val_scaled, val_df[["segmentation_failed"]].values])

X_train_df = pd.DataFrame(X_train_full, columns=feature_names)
X_val_df = pd.DataFrame(X_val_full, columns=feature_names)


pd.DataFrame(X_train_df, columns=feature_names).to_csv("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/train_features.csv", index=False)

# === Train XGBoost ===
ratio = sum(y_train == 0) / sum(y_train == 1)
model = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=1,
    n_estimators=500,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='auc', 
    max_delta_step=1
)

model.fit(X_train_df, y_train)

# Plot top N features (you can adjust `max_num_features`)
plt.figure(figsize=(8, 6))
plot_importance(model, importance_type='gain', max_num_features=12, xlabel="Gain")
plt.title("XGBoost Feature Importance (Gain)")
plt.tight_layout()
plt.savefig("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/feature_importance_gain.png")
plt.show()

explainer = shap.Explainer(model, X_train_df)
shap_values = explainer(X_val_df)

# === Global Summary Plot ===
shap.plots.beeswarm(shap_values, max_display=12)  # top 12 features
plt.savefig("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/shap_beeswarm.png")
plt.show()

# === Bar Plot of Mean SHAP Values ===
shap.plots.bar(shap_values, max_display=12)
plt.savefig("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/shap_bar.png")
plt.show()


# === Save model and scaler ===
joblib.dump(model, "/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/xgb_pcr_model.pkl")
joblib.dump(scaler, "/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/scaler.pkl")
print("✅ Model and scaler saved.")

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

df_preds = pd.DataFrame({
    "patient_id": ids_val,
    "true_label": y_val,
    "pred_label": y_pred,
    "prob_class_1": y_prob
})
df_preds.to_csv("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/validation_predictions.csv", index=False)
print("📁 Predictions saved to validation_predictions.csv")