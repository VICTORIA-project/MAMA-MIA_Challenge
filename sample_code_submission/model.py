# ==================== MAMA-MIA CHALLENGE SAMPLE SUBMISSION ====================
#
# This is the official sample submission script for the **MAMA-MIA Challenge**, 
# covering both tasks:
#
#   1. Primary Tumour Segmentation (Task 1)
#   2. Treatment Response Classification (Task 2)
#
# ----------------------------- SUBMISSION FORMAT -----------------------------
# Participants must implement a class `Model` with one or two of these methods:
#
#   - `predict_segmentation(output_dir)`: required for Task 1
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#
#   - `predict_classification(output_dir)`: required for Task 2
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
#   - `predict_classification(output_dir)`: if a single model handles both tasks
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
# You can submit:
#   - Only Task 1 (implement `predict_segmentation`)
#   - Only Task 2 (implement `predict_classification`)
#   - Both Tasks (implement both methods independently or define `predict_segmentation_and_classification` method)
#
# ------------------------ SANITY-CHECK PHASE ------------------------
#
# ✅ Before entering the validation or test phases, participants must pass the **Sanity-Check phase**.
#   - This phase uses **4 samples from the test set** to ensure your submission pipeline runs correctly.
#   - Submissions in this phase are **not scored**, but must complete successfully within the **20-minute timeout limit**.
#   - Use this phase to debug your pipeline and verify output formats without impacting your submission quota.
#
# 💡 This helps avoid wasted submissions on later phases due to technical errors.
#
# ------------------------ SUBMISSION LIMITATIONS ------------------------
#
# ⚠️ Submission limits are strictly enforced per team:
#   - **One submission per day**
#   - **Up to 15 submissions total on the validation set**
#   - **Only 1 final submission on the test set**
#
# Plan your development and testing accordingly to avoid exhausting submissions prematurely.
#
# ----------------------------- RUNTIME AND RESOURCES -----------------------------
#
# > ⚠️ VERY IMPORTANT: Each image has a **timeout of 5 minutes** on the compute worker.
#   - **Validation Set**: 58 patients → total budget ≈ 290 minutes
#   - **Test Set**: 516 patients → total budget ≈ 2580 minutes
#
# > The compute worker environment is based on the Docker image:
#       `lgarrucho/codabench-gpu:latest`
#
# > You can install additional dependencies via `requirements.txt`.
#   Please ensure all required packages are listed there.
#
# ----------------------------- SEGMENTATION DETAILS -----------------------------
#
# This example uses `nnUNet v2`, which is compatible with the GPU compute worker.
# Note the following nnUNet-specific constraints:
#
# ✅ `predict_from_files_sequential` MUST be used for inference.
#     - This is because nnUNet’s multiprocessing is incompatible with the compute container.
#     - In our environment, a single fold prediction using `predict_from_files_sequential` 
#       takes approximately **1 minute per patient**.
#
# ✅ The model uses **fold 0 only** to reduce runtime.
# 
# ✅ Predictions are post-processed by applying a breast bounding box mask using 
#    metadata provided in the per-patient JSON file.
#
# ----------------------------- CLASSIFICATION DETAILS -----------------------------
#
# If using predicted segmentations for Task 2 classification:
#   - Save them in `self.predicted_segmentations` inside `predict_segmentation()`
#   - You can reuse them in `predict_classification()`
#   - Or perform Task 1 and Task 2 inside `predict_segmentation_and_classification`
#
# ----------------------------- DATASET INTERFACE -----------------------------
# The provided `dataset` object is a `RestrictedDataset` instance and includes:
#
#   - `dataset.get_patient_id_list() → list[str]`  
#         Patient IDs for current split (val/test)
#
#   - `dataset.get_dce_mri_path_list(patient_id) → list[str]`  
#         Paths to all image channels (typically pre and post contrast)
#         - iamge_list[0] corresponds to the pre-contrast image path
#         - iamge_list[1] corresponds to the first post-contrast image path and so on
#
#   - `dataset.read_json_file(patient_id) → dict`  
#         Metadata dictionary per patient, including:
#         - breast bounding box (`primary_lesion.breast_coordinates`)
#         - scanner metadata (`imaging_data`), etc...
#
# Example JSON structure:
# {
#   "patient_id": "XXX_XXX_SXXXX",
#   "primary_lesion": {
#     "breast_coordinates": {
#         "x_min": 1, "x_max": 158,
#         "y_min": 6, "y_max": 276,
#         "z_min": 1, "z_max": 176
#     }
#   },
#   "imaging_data": {
#     "bilateral": true,
#     "dataset": "HOSPITAL_X",
#     "site": "HOSPITAL_X",
#     "scanner_manufacturer": "SIEMENS",
#     "scanner_model": "Aera",
#     "field_strength": 1.5,
#     "echo_time": 1.11,
#     "repetition_time": 3.35
#   }
# }
#
# ----------------------------- RECOMMENDATIONS -----------------------------
# ✅ We recommend to always test your submission first in the Sanity-Check Phase.
#    As in Codabench the phases need to be sequential and they cannot run in parallel,
#    we will open a secondary MAMA-MIA Challenge Codabench page with a permanen Sanity-Check phase.
#   That way you won't lose submission trials to the validation or even wore, the test set.
# ✅ We recommend testing your solution locally and measuring execution time per image.
# ✅ Use lightweight models or limit folds if running nnUNet.
# ✅ Keep all file paths, patient IDs, and formats **exactly** as specified.
# ✅ Ensure your output folders are created correctly (e.g. `pred_segmentations/`)
# ✅ For faster runtime, only select a single image for segmentation.
#
# ------------------------ COPYRIGHT ------------------------------------------
#
# © 2025 Lidia Garrucho. All rights reserved.
# Unauthorized use, reproduction, or distribution of any part of this competition's 
# materials is prohibited without explicit permission.
#
# ------------------------------------------------------------------------------

# === MANDATORY IMPORTS ===
import os
import pandas as pd
import shutil

# === OPTIONAL IMPORTS: only needed if you modify or extend nnUNet input/output handling ===
# You can remove unused imports above if not needed for your solution
import numpy as np
import torch
import SimpleITK as sitk
# === nnUNetv2 IMPORTS ===
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import radiomics
import logging

# Suppress non-critical logs
radiomics.logger.setLevel(logging.ERROR)
logging.getLogger("radiomics").setLevel(logging.ERROR)

import json
from tqdm import tqdm
import joblib
from scipy.ndimage import binary_erosion
# from xgboost import XGBClassifier
# from sklearn.preprocessing import StandardScaler
from radiomics import featureextractor


class Model:
    def __init__(self, dataset):
        """
        Initializes the model with the restricted dataset.
        
        Args:
            dataset (RestrictedDataset): Preloaded dataset instance with controlled access.
        """
        # MANDATOR
        self.dataset = dataset  # Restricted Access to Private Dataset
        self.predicted_segmentations = None  # Optional: stores path to predicted segmentations
        # Only if using nnUNetv2, you can define here any other variables
        self.dataset_id = "108"  # Dataset ID must match your folder structure
        self.config = "3d_fullres" # nnUNetv2 configuration
        

# IMPORTANT: The definition of this method will skip the execution of `predict_segmentation` and `predict_classification` if defined
    def predict_segmentation_and_classification(self, output_dir):
        """
        Define this method if your model performs both Task 1 (segmentation) and Task 2 (classification).
        
        This naive combined implementation:
            - Generates segmentation masks using thresholding.
            - Applies a rule-based volume threshold for response classification.
        
        Args:
            output_dir (str): Path to the output directory.
        
        Returns:
            str: Path to the directory containing the predicted segmentation masks (Task 1).
            DataFrame: Pandas DataFrame containing predicted labels and scores (Task 2).
        """
        
        # === Set required nnUNet paths ===
        # Not strictly mandatory if pre-set in Docker env, but avoids missing variable warnings
        os.environ['nnUNet_raw'] = "/app/ingested_program/sample_code_submission"
        os.environ['nnUNet_preprocessed'] = "/app/ingested_program/sample_code_submission"
        os.environ['nnUNet_results'] = "/app/ingested_program/sample_code_submission"

        # Usage: https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/inference
        # === Instantiate nnUNet Predictor ===
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        # === Load your trained model from a specific fold ===
        predictor.initialize_from_trained_model_folder(
            '/app/ingested_program/sample_code_submission/Dataset108_multiC/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres',
            use_folds=(0,), checkpoint_name='checkpoint_final.pth')
        
        # === Build nnUNet-compatible input images folder ===
        nnunet_input_images = os.path.join(output_dir, 'nnunet_input_images')
        os.makedirs(nnunet_input_images, exist_ok=True)
        
        
        def resample_to_spacing(image, output_spacing, interpolator=sitk.sitkNearestNeighbor):
            original_size = image.GetSize()
            original_spacing = image.GetSpacing()
            original_origin = image.GetOrigin()
            original_direction = image.GetDirection()

            output_size = [
                int(round(osz * ospc / nspc)) 
                for osz, ospc, nspc in zip(original_size, original_spacing, output_spacing)
            ]

            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing(output_spacing)
            resample.SetSize(output_size)
            resample.SetOutputDirection(original_direction)
            resample.SetOutputOrigin(original_origin)
            resample.SetInterpolator(interpolator)

            return resample.Execute(image)

        def clip_image_sitk(image_sitk, percentiles=[0.1, 99.9]):
            image_array = sitk.GetArrayFromImage(image_sitk).ravel()
            image_array = image_array[image_array != 0]
            if len(image_array) == 0:
                return image_sitk
            lowerbound = np.percentile(image_array, percentiles[0])
            upperbound = np.percentile(image_array, percentiles[1])
            clamp_filter = sitk.ClampImageFilter()
            clamp_filter.SetLowerBound(float(lowerbound))
            clamp_filter.SetUpperBound(float(upperbound))
            return clamp_filter.Execute(image_sitk)

        def zscore_normalization_sitk(image_sitk, mean, std):
            array = sitk.GetArrayFromImage(image_sitk)
            normalized_array = (array - mean) / std
            normalized_sitk = sitk.GetImageFromArray(normalized_array)
            normalized_sitk.CopyInformation(image_sitk)
            return normalized_sitk

        def crop_image_with_coords(image_path, coords):
            img_sitk = sitk.ReadImage(image_path, sitk.sitkFloat32)
            img_np = sitk.GetArrayFromImage(img_sitk)
            shape = img_np.shape  # (Z, Y, X)

            x_min, x_max = coords["x_min"], coords["x_max"]
            y_min, y_max = coords["y_min"], coords["y_max"]
            z_min, z_max = coords["z_min"], coords["z_max"]

            if not (0 <= x_min < x_max <= shape[0] and
                    0 <= y_min < y_max <= shape[1] and
                    0 <= z_min < z_max <= shape[2]):
                raise ValueError("Coordinates out of bounds.")

            cropped_np = img_np[x_min:x_max, y_min:y_max, z_min:z_max]
            cropped_sitk = sitk.GetImageFromArray(cropped_np)

            spacing = img_sitk.GetSpacing()
            direction = img_sitk.GetDirection()
            origin_phys  = img_sitk.TransformIndexToPhysicalPoint((z_min, y_min, x_min))

            coords["origin_phys"] = origin_phys
            coords["original_spacing"] = spacing

            cropped_sitk.SetSpacing(spacing)
            cropped_sitk.SetDirection(direction)
            cropped_sitk.SetOrigin(origin_phys )

            return cropped_sitk, coords
        
        def restore_mask_to_original_space_from_data(
            patient_id,
            cropped_mask_path,
            original_image_path,
            coords,
            output_path
        ):
            """
            Restores a cropped mask to its original spatial size using provided bounding box coordinates.
            """

            # === Load original image to get shape and metadata ===
            original_sitk = sitk.ReadImage(original_image_path)

            # Prepare an empty full-size mask
            restored_sitk = sitk.Image(original_sitk.GetSize(), sitk.sitkUInt8)
            restored_sitk.CopyInformation(original_sitk)

            # === Load cropped predicted mask ===
            cropped_mask_sitk = sitk.ReadImage(cropped_mask_path, sitk.sitkUInt8)

            # === Resample to original spacing BEFORE setting origin
            resampled_pred = resample_to_spacing(cropped_mask_sitk, coords["original_spacing"])

            # === Now set the correct origin
            origin_phys = coords["origin_phys"]
            resampled_pred.SetOrigin(origin_phys)

            # Use physical origin to compute insertion index
            destination_index = original_sitk.TransformPhysicalPointToIndex(origin_phys)

            # Paste into full-size mask
            restored_sitk = sitk.Paste(
                destinationImage=restored_sitk,
                sourceImage=resampled_pred,
                sourceSize=resampled_pred.GetSize(),
                sourceIndex=[0, 0, 0],
                destinationIndex=destination_index
            )

            # === Save result ===
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(restored_sitk, output_path, useCompression=True)
        
        # === Main Loop ===
        patient_coords = {}  # Add this line before the loop
        # === Participants can modify how they prepare input ===
        patient_ids = self.dataset.get_patient_id_list()
        for patient_id in patient_ids:
            patient_info = self.dataset.read_json_file(patient_id) # hadeel
            if not patient_info or "primary_lesion" not in patient_info:
                continue

            coords = patient_info["primary_lesion"]["breast_coordinates"]
            
            images = self.dataset.get_dce_mri_path_list(patient_id) # image_paths
            
            try:
                if len(images) < 3:
                    print(f"⚠️ Not enough phases for {patient_id}, skipping.")
                    continue

                # Crop only [0], [1], [2]
                results = [crop_image_with_coords(images[i], coords) for i in [0, 1, 2]]
                cropped_images = [res[0] for res in results]         # Extract all cropped images
                updated_coords = results[0][1]  

                patient_coords[patient_id] = updated_coords  # ✅ Save per patient

                if len(cropped_images) != 3:
                    print(f"⚠️ Incomplete cropping for {patient_id}, skipping.")
                    continue

                # Compute mean/std from pre-contrast (index 0)
                pre_array = sitk.GetArrayFromImage(cropped_images[0])
                mean, std = np.mean(pre_array), np.std(pre_array)
                if std == 0:
                    print(f"⚠️ Std = 0 for {patient_id}, skipping.")
                    continue
                # Process and save each of the 3 channels
                for channel_index, cropped_img in zip([0, 1, 2], cropped_images):
                    clipped = clip_image_sitk(cropped_img)
                    normalized = zscore_normalization_sitk(clipped, mean, std)
                    resampled = resample_to_spacing(normalized, [1.0, 1.0, 1.0], interpolator=sitk.sitkBSpline)
                    # Save to preprocessed folder
                    nnunet_image_path = os.path.join(nnunet_input_images, f"{patient_id}_{channel_index:04d}_0000.nii.gz")
                    sitk.WriteImage(resampled, nnunet_image_path, useCompression=True)

                    # print(f"Saved channel {channel_index} for {patient_id} → {nnunet_image_path}")


            except Exception as e:
                print(f"Failed processing {patient_id}: {e}")

        # === Output folder for raw nnUNet segmentations ===
        output_dir_nnunet = os.path.join(output_dir, 'nnunet_seg')
        os.makedirs(output_dir_nnunet, exist_ok=True)

        # === Call nnUNetv2 prediction ===
        nnunet_images = []
        for patient_id in patient_ids:
            input_paths = [
                os.path.join(nnunet_input_images, f"{patient_id}_{i:04d}_0000.nii.gz")
                for i in [0, 1, 2]
            ]
            if all(os.path.exists(p) for p in input_paths):
                nnunet_images.append(input_paths)
            else:
                print(f"❌ Skipping inference for {patient_id}, missing one or more input channels.")

        # === Run Prediciton ===
        # IMPORTANT: the only method that works inside the Docker container is predict_from_files_sequential
        # This method will predict all images in the list and save them in the output directory
        ret = predictor.predict_from_files_sequential(nnunet_images, output_dir_nnunet, save_probabilities=False,
                                                       overwrite=True, folder_with_segs_from_prev_stage=None)
        print("Predictions saved to:", os.listdir(output_dir_nnunet))
        
        # Folder to store predicted segmentation masks
        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        os.makedirs(output_dir_final, exist_ok=True)

        # === Restoring the original spatial space === # hadeel

        for patient_id in self.dataset.get_patient_id_list():
            seg_path = os.path.join(output_dir_nnunet, f"{patient_id}_0000.nii.gz")
            if not os.path.exists(seg_path):
                print(f'{seg_path} NOT FOUND!')
                continue
            
            # === Load bounding box from JSON via dataset
            patient_info = self.dataset.read_json_file(patient_id)
            if not patient_info or "primary_lesion" not in patient_info:
                print(f"⚠️ No primary lesion info for {patient_id}, skipping.")
                continue
            
            # === Use first post-contrast image as shape/metadata reference
            original_image_path = self.dataset.get_dce_mri_path_list(patient_id)[1]
                        
            # MANDATORY: the segmentation masks should be named using the patient_id
            final_seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
            
            try:
                restore_mask_to_original_space_from_data(
                    patient_id,
                    cropped_mask_path=seg_path,
                    original_image_path=original_image_path,
                    coords= patient_coords.get(patient_id),
                    output_path=final_seg_path
                )
                print(f"✅ Restored and saved: {final_seg_path}")

                # === Delete preprocessed files ===
                for channel_index in [0, 1, 2]:
                    nnunet_image_path = os.path.join(nnunet_input_images, f"{patient_id}_{channel_index:04d}_0000.nii.gz")
                    if os.path.exists(nnunet_image_path):
                        os.remove(nnunet_image_path)
                        # print(f"🗑️ Deleted nnUNet input file: {nnunet_image_path}")

            except Exception as e:
                print(f"⚠️ Restoration failed for {patient_id}: {e}")
            

        # === Load all fold models ===
        fold_models = []
        for i in range(5):
            model_path = f"/app/ingested_program/sample_code_submission/ML/fold_{i}/xgb_model.pkl"
            fold_models.append(joblib.load(model_path))

        predictions = []
        

        def normalize_nonzero(image_np):
            nonzero = image_np[image_np != 0]
            if nonzero.size == 0:
                return image_np
            mean = np.mean(nonzero)
            std = np.std(nonzero)
            return (image_np - mean) / std

        # --- 1. Feature extraction returns a dict (no NaNs) ---

        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableFeatureClassByName('shape')
        extractor.settings['force2D'] = False

        radiomics_shape_feature_names = [
            "Elongation", "Flatness", "LeastAxisLength", "MajorAxisLength",
            "Maximum2DDiameterColumn", "Maximum2DDiameterRow", "Maximum2DDiameterSlice",
            "Maximum3DDiameter", "MeshVolume", "MinorAxisLength", "Sphericity",
            "SurfaceArea", "SurfaceVolumeRatio", "VoxelVolume"
        ]

        def extract_features_multiphase(image_paths, mask_path):
            mask = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask).astype(bool)
            image_for_shape = sitk.ReadImage(image_paths[1])  # Use _0001.nii.gz for radiomics

            # Call PyRadiomics with resampled image and mask (not mask path!)
            try:
                result = extractor.execute(image_for_shape, mask) 
                shape_feats = {
                    k.split("_")[-1]: v for k, v in result.items()
                    if k.startswith("original_shape_")
                }
            except Exception as e:
                print(f"❌ Radiomics failed on {image_paths[1]} | Error: {e}")
                shape_feats = {feat: 0 for feat in radiomics_shape_feature_names}

            if np.sum(mask_array) == 0 or all(v == 0 for v in shape_feats.values()):
                return {
                    "mean_baseline": 0, "peak": 0, "auc": 0, "range": 0,
                    "wash_in": 0, "wash_out": 0, "voxel_mean_over_time": 0,
                    "segmentation_failed": 1,
                    **{feat: 0 for feat in radiomics_shape_feature_names}
                }

            dce_volumes = []
            for img_path in image_paths:
                img = sitk.ReadImage(img_path)
                img_array = sitk.GetArrayFromImage(img)
                clipped = np.clip(img_array, np.percentile(img_array, 0.5), np.percentile(img_array, 99.5))
                normed = normalize_nonzero(clipped)
                dce_volumes.append(normed)
            dce_4d = np.stack(dce_volumes, axis=-1)

            intensity_curve = []
            for t in range(dce_4d.shape[-1]):
                phase = dce_4d[..., t]
                roi_values = phase[mask_array]
                intensity_curve.append(np.mean(roi_values))
            intensity_curve = np.array(intensity_curve)

            baseline = intensity_curve[0]
            peak = np.max(intensity_curve)
            time_to_peak = np.argmax(intensity_curve)
            auc = np.sum(intensity_curve)
            wash_in = (peak - baseline) / (time_to_peak + 1e-6)
            wash_out = (peak - intensity_curve[-1]) / (len(intensity_curve) - time_to_peak + 1e-6)
            intensity_range = peak - np.min(intensity_curve)
            voxel_mean_over_time = dce_4d[mask_array].mean()

            features = {
                "mean_baseline": baseline,
                "peak": peak,
                "auc": auc,
                "range": intensity_range,
                "wash_in": wash_in,
                "wash_out": wash_out,
                "voxel_mean_over_time": voxel_mean_over_time,
                "segmentation_failed": 0,
                **shape_feats
            }

            # Convert all feature values to float (safely)
            for k, v in features.items():
                try:
                    features[k] = float(v)
                except:
                    features[k] = 0.0  # fallback if parsing fails

            return features


        feature_names = [
            "mean_baseline", "peak", "auc", "range", "wash_in", "wash_out",
            "voxel_mean_over_time", "segmentation_failed",
            "Elongation", "Flatness", "LeastAxisLength", "MajorAxisLength",
            "Maximum2DDiameterColumn", "Maximum2DDiameterRow", "Maximum2DDiameterSlice",
            "Maximum3DDiameter", "MeshVolume", "MinorAxisLength", "Sphericity",
            "SurfaceArea", "SurfaceVolumeRatio", "VoxelVolume"
        ]


        for patient_id in self.dataset.get_patient_id_list():
            # Load DCE-MRI series (assuming post-contrast is the second timepoint)
            image_paths = self.dataset.get_dce_mri_path_list(patient_id)
            if not image_paths or len(image_paths) < 2:
                continue

            # if self.predicted_segmentations:
            # === Example using segmentation-derived feature (volume) ===
            seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
            if not os.path.exists(seg_path):
                continue
            
            features = extract_features_multiphase(image_paths, seg_path)
            
            # After creating DataFrame
            input_df = pd.DataFrame([features])[feature_names]
            input_df = input_df.astype(float).fillna(0)  # enforce numeric, remove NaNs

            # Check if fallback is needed
            if features["segmentation_failed"] == 1:
                y_prob_avg = 0.5
                y_pred_final = 0
            else:
                # === Ensemble inference ===
                probs = [model.predict_proba(input_df)[0, 1] for model in fold_models]
                y_prob_avg = np.mean(probs)
                y_pred_final = int(y_prob_avg >= 0.5)

            predictions.append({
                "patient_id": patient_id,
                "pcr": y_pred_final,
                "score": y_prob_avg
            })

        return output_dir_final, pd.DataFrame(predictions)