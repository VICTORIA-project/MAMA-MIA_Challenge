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

# Load JSON
# with open("/home/hadeel/MAMA-MIA_Challenge/train_val_split_fold0_reformatted.json", "r") as f:
#     data = json.load(f)

# Correct usage
# extractor = featureextractor.RadiomicsFeatureExtractor()
# extractor.disableAllFeatures()
# # extractor.enableAllFeatures()
# extractor.enableFeatureClassByName('shape')
# extractor.settings['force2D'] = True


# === Extract from one sample ===
# def extract_shape_features(image_path, mask_path):
#     try:
#         result = extractor.execute(image_path, mask_path)
#         shape_feats = {
#             k.replace("original_shape_", ""): v
#             for k, v in result.items()
#             if k.startswith("original_shape_")
#         }
#         return shape_feats
#     except Exception as e:
#         print(f"❌ Failed to extract features from {image_path}")
#         print(f"    Mask: {mask_path}")
#         print(f"    Error: {e}")
#         return None


def extract_features(img_path, mask_path):
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)

    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)

    if np.sum(mask_array > 0) == 0:
        print("Warning: Empty mask. Returning default features.")
        return {
            "volume_vox": 0,
            "surface_vox": 0,
            "bbox_volume": 0,
            "extent": np.nan,
            "compactness": np.nan,
            "segmentation_failed": 1
        } 

    # Intensity features
    # mean = roi.mean()
    # std = roi.std()
    # min_val = roi.min()
    # max_val = roi.max()
    # p25 = np.percentile(roi, 25)
    # p75 = np.percentile(roi, 75)
    
    volume_vox = np.sum(mask_array > 0)

    # Surface and shape features
    eroded = binary_erosion(mask_array)
    surface_vox = np.sum(mask_array != eroded)

    coords = np.argwhere(mask_array)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_dims = bbox_max - bbox_min + 1
    bbox_volume = np.prod(bbox_dims)

    # elongation = bbox_dims.max() / max(1, bbox_dims.min())
    compactness = (volume_vox ** 2) / max(1, surface_vox ** 3)
    # sphericity = (np.pi ** (1/3) * (6 * volume_vox) ** (2/3)) / max(1, surface_vox)
    
    extent = volume_vox / max(1, bbox_volume)
   
    # try:
    #     hull = ConvexHull(coords)
    #     convex_volume = hull.volume
    #     solidity = volume_vox / max(1, convex_volume)
    # except Exception:
    #     convex_volume = 0
    #     solidity = 0  # fallback if hull fails (e.g. too few points)

    # equivalent_diameter = (6 * volume_vox / np.pi) ** (1 / 3)
    
    # coords_centered = coords - coords.mean(axis=0)
    # cov = np.cov(coords_centered, rowvar=False)
    # eigvals = np.linalg.eigvalsh(cov)  # sorted ascending

    # flatness: ratio of smallest to largest axis variance
    # flatness = eigvals[0] / max(1e-5, eigvals[2])  # to avoid div by zero
    
    # major_axis_length = np.sqrt(eigvals[2])
    # minor_axis_length = np.sqrt(eigvals[0])
    
    # com = coords.mean(axis=0)  # [z, y, x]
    # com_z, com_y, com_x = com
    
    # _, num_components = label(mask_array)


    return [
        #mean, std, min_val, max_val, p25, p75,
        volume_vox,
        surface_vox, 
        bbox_volume,
        # elongation ,
        compactness, 
        # sphericity,
        extent,
        segmentation_failed
        # solidity,
        # equivalent_diameter,
        # flatness,
        # major_axis_length,
        # minor_axis_length,
        # com_z, com_y, com_x,
        # num_components
    ]


feature_names = [
    #"mean", "std", "min_val", "max_val", "p25", "p75", 
    "volume_vox",
    "surface_vox", 
    "bbox_volume",
    # "elongation", 
    "compactness",
    # "sphericity",
    "extent",
    "segmentation_failed"
    # "solidity",
    # "equivalent_diameter",
    # "flatness",
    # "major_axis_length",
    # "minor_axis_length",
    # "com_z", "com_y", "com_x",
    # "num_components"

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
            X.append(feats)
            y.append(label)
            ids.append(pid)
    return np.array(X), np.array(y), ids

# === Wrapper to extract dataset ===
# def load_data(entries, feature_names=None):
#     X, y, ids = [], [], []

#     for entry in entries:
#         image_path = entry["image"]
#         if isinstance(image_path, list):
#             image_path = image_path[0]
#         mask_path = entry["mask"]
#         label = entry["pcr"]
#         case_id = os.path.basename(mask_path).split(".")[0]

#         feats = extract_shape_features(image_path, mask_path)
#         if feats is None:
#             continue

#         if feature_names is None:
#             feature_names = list(feats.keys())

#         # Consistency check
#         feat_vector = [feats.get(k, 0) for k in feature_names]
#         X.append(feat_vector)
#         y.append(label)
#         ids.append(case_id)

#     return np.array(X), np.array(y), ids, feature_names

# === Load JSON ===
with open("train_val_split_fold0_reformatted.json", "r") as f:
    data = json.load(f)

train_entries = data["fold_0"]["train"]
val_entries = data["fold_0"]["val"]

# # Training features
# print("📦 Extracting training features...")
# X_train, y_train, ids_train, feature_names = load_data(train_entries)
# pd.DataFrame(X_train, columns=feature_names).to_csv(
#     "/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/train_features_radiomics.csv", index=False)

# # Validation features
# print("📦 Extracting validation features...")
# X_val, y_val, ids_val, _ = load_data(val_entries, feature_names=feature_names)
# pd.DataFrame(X_val, columns=feature_names).to_csv(
#     "/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/val_features_radiomics.csv", index=False)

# === Extract features ===
print("Extracting training features...")
X_train, y_train, ids_train = load_data(train_entries)

pd.DataFrame(X_train, columns=feature_names).to_csv("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/train_features.csv", index=False)

print("Extracting validation features...")
X_val, y_val, ids_val = load_data(val_entries)

# === Normalize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Univariate Feature Selection
# selector = SelectKBest(score_func=mutual_info_classif, k=4)  # or use mutual_info_classif
# X_train_kbest = selector.fit_transform(X_train_scaled, y_train)
# X_val_kbest = selector.transform(X_val_scaled)

# # Get selected feature names
# selected_mask = selector.get_support()
# selected_features = [f for f, keep in zip(feature_names, selected_mask) if keep]

pd.DataFrame(X_train_scaled, columns=feature_names).to_csv("/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/train_features.csv", index=False)

# pd.DataFrame(X_val_kbest, columns=selected_features).to_csv(
#     "/home/hadeel/MAMA-MIA_Challenge/machine_learning/outputs/val_features_radiomics.csv", index=False)

# === Train XGBoost ===
ratio = sum(y_train == 0) / sum(y_train == 1)
model = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=ratio,
    n_estimators=500,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='auc', 
    max_delta_step=0,
    # device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
)
# model = XGBRFClassifier(n_estimators=500, subsample=0.8, colsample_bynode=0.8, random_state=42, scale_pos_weight=ratio, max_depth=5)
# model = XGBRFClassifier(objective="binary:logistic",
#     scale_pos_weight=ratio,
#     n_estimators=500,
#     max_depth=5,
#     use_label_encoder=False,
#     eval_metric='auc', 
#     max_delta_step=0, 
#     subsample=0.8, colsample_bynode=0.8, random_state=42
#     )


X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_val_df = pd.DataFrame(X_val_scaled, columns=feature_names)

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

# Predict using the model
y_pred_all = model.predict(X_val_scaled)
y_prob_all = model.predict_proba(X_val_scaled)[:, 1]

# Initialize with model predictions
y_pred = y_pred_all.copy()
y_prob = y_prob_all.copy()

# Apply fallback for segmentation failures
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
