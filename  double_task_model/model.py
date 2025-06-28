# ==================== MAMA-MIA CHALLENGE SAMPLE SUBMISSION ====================
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

import json
from tqdm import tqdm
import joblib
from sklearn.metrics import (
    classification_report, roc_auc_score, balanced_accuracy_score, roc_curve
)
from scipy.ndimage import binary_erosion
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


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
        self.dataset_id = "105"  # Dataset ID must match your folder structure
        self.config = "3d_fullres" # nnUNetv2 configuration
        

    def predict_segmentation(self, output_dir):
        """
        Task 1 — Predict tumor segmentation with nnUNetv2.
        You MUST define this method if participating in Task 1.

        Args:
            output_dir (str): Directory where predictions will be stored.

        Returns:
            str: Path to folder with predicted segmentation masks.
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
            '/app/ingested_program/sample_code_submission/Dataset105_Breast/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres',
            use_folds=(0,), checkpoint_name='checkpoint_final.pth')
        
        # === Build nnUNet-compatible input images folder ===
        nnunet_input_images = os.path.join(output_dir, 'nnunet_input_images')
        os.makedirs(nnunet_input_images, exist_ok=True)

        # === Participants can modify how they prepare input ===
        patient_ids = self.dataset.get_patient_id_list()
        for patient_id in patient_ids:
            images = self.dataset.get_dce_mri_path_list(patient_id)
            # Select the image or images to be used to predict the final segmentation
            # For example, using only the first post-contrast image
            first_post_contrast_image = images[1]
            # Save the image in the nnUNet format (ending in _0000.nii.gz)
            nnunet_image_path = os.path.join(nnunet_input_images, f"{patient_id}_0001_0000.nii.gz")
            # Copy and rename the image to the nnUNet format
            shutil.copy(first_post_contrast_image, nnunet_image_path)

        # === Output folder for raw nnUNet segmentations ===
        output_dir_nnunet = os.path.join(output_dir, 'nnunet_seg')
        os.makedirs(output_dir_nnunet, exist_ok=True)

        # === Call nnUNetv2 prediction ===
        nnunet_images = [[os.path.join(nnunet_input_images, f)] for f in os.listdir(nnunet_input_images)]
        # IMPORTANT: the only method that works inside the Docker container is predict_from_files_sequential
        # This method will predict all images in the list and save them in the output directory
        ret = predictor.predict_from_files_sequential(nnunet_images, output_dir_nnunet, save_probabilities=False,
                                                       overwrite=True, folder_with_segs_from_prev_stage=None)
        print("Predictions saved to:", os.listdir(output_dir_nnunet))
        
       # === Final output folder (MANDATORY name) ===
        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        os.makedirs(output_dir_final, exist_ok=True)

        # === Optional post-processing step ===
        # For example, you can threshold the predictions or apply morphological operations
        # Here, we iterate through the predicted segmentations and apply the breast mask to each segmentation
        # to remove false positives outside the breast region
        for patient_id in self.dataset.get_patient_id_list():
            seg_path = os.path.join(output_dir_nnunet, f"{patient_id}_0001.nii.gz")
            if not os.path.exists(seg_path):
                print(f'{seg_path} NOT FOUND!')
                continue
            
            segmentation = sitk.ReadImage(seg_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            
            patient_info = self.dataset.read_json_file(patient_id)
            if not patient_info or "primary_lesion" not in patient_info:
                continue
            
            coords = patient_info["primary_lesion"]["breast_coordinates"]
            x_min, x_max = coords["x_min"], coords["x_max"]
            y_min, y_max = coords["y_min"], coords["y_max"]
            z_min, z_max = coords["z_min"], coords["z_max"]
            
            masked_segmentation = np.zeros_like(segmentation_array)
            masked_segmentation[x_min:x_max, y_min:y_max, z_min:z_max] = \
                segmentation_array[x_min:x_max, y_min:y_max, z_min:z_max]            
            masked_seg_image = sitk.GetImageFromArray(masked_segmentation)
            masked_seg_image.CopyInformation(segmentation)

            # MANDATORY: the segmentation masks should be named using the patient_id
            final_seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
            sitk.WriteImage(masked_seg_image, final_seg_path)

        # Save path for Task 2 if needed
        self.predicted_segmentations = output_dir_final

        return output_dir_final
    
    # def predict_classification(self, output_dir):
    #     """
    #     Task 2 — Predict treatment response (pCR).
    #     You MUST define this method if participating in Task 2.

    #     Args:
    #         output_dir (str): Directory to save output predictions.

    #     Returns:
    #         pd.DataFrame: DataFrame with patient_id, pcr prediction, and score.
    #     """
    #     patient_ids = self.dataset.get_patient_id_list()
    #     predictions = []
        
    #     for patient_id in patient_ids:
    #         if self.predicted_segmentations:
    #             # === Example using segmentation-derived feature (volume) ===
    #             seg_path = os.path.join(self.predicted_segmentations, f"{patient_id}.nii.gz")
    #             if not os.path.exists(seg_path):
    #                 continue
                
    #             segmentation = sitk.ReadImage(seg_path)
    #             segmentation_array = sitk.GetArrayFromImage(segmentation)
    #             # You can use the predicted segmentation to compute features if task 1 is done
    #             # For example, compute the volume of the segmented region
    #             # ...

    #             # RANDOM CLASSIFIER AS EXAMPLE
    #             # Replace with real feature extraction + ML model
    #             probability = np.random.rand()
    #             pcr_prediction = int(probability > 0.5)

    #         else:
    #             # === Example using raw image intensity for rule-based prediction ===
    #             image_paths = self.dataset.get_dce_mri_path_list(patient_id)
    #             if not image_paths:
    #                 continue
                
    #             image = sitk.ReadImage(image_paths[1])
    #             image_array = sitk.GetArrayFromImage(image)
    #             mean_intensity = np.mean(image_array)
    #             pcr_prediction = 1 if mean_intensity > 500 else 0
    #             probability = np.random.rand() if pcr_prediction == 1 else np.random.rand() / 2
            
    #         # === MANDATORY output format ===
    #         predictions.append({
    #             "patient_id": patient_id,
    #             "pcr": pcr_prediction,
    #             "score": probability
    #         })

    #     return pd.DataFrame(predictions)
    
    # def is_empty_mask(mask_path):
    #     mask = sitk.ReadImage(mask_path)
    #     mask_np = sitk.GetArrayFromImage(mask)
    #     return np.sum(mask_np) == 0

    # def normalize_nonzero(image_np):
    #     nonzero = image_np[image_np != 0]
    #     if len(nonzero) == 0:
    #         return image_np  # avoid division by zero
    #     mean = nonzero.mean()
    #     std = nonzero.std()
    #     return (image_np - mean) / std

    # def load_and_crop_image(image_path, coords):
    #     # Load image
    #     image_sitk = sitk.ReadImage(image_path)
    #     image_np = sitk.GetArrayFromImage(image_sitk)  # Shape: (D, H, W)

    #     # Normalize
    #     image_np = normalize_nonzero(image_np)

    #     # Extract bounding box coordinates
    #     x_min, x_max = coords["x_min"], coords["x_max"]
    #     y_min, y_max = coords["y_min"], coords["y_max"]
    #     z_min, z_max = coords["z_min"], coords["z_max"]

    #     # Crop using NumPy (Note: SITK uses (z, y, x) indexing)
    #     cropped_np = image_np[z_min:z_max, y_min:y_max, x_min:x_max]

    #     # Add channel dimension (as MONAI's EnsureChannelFirstd would)
    #     cropped_np = cropped_np[np.newaxis, :, :, :]  # Shape: (C, D, H, W)

    #     # Convert to torch.Tensor
    #     cropped_tensor = torch.from_numpy(cropped_np.astype(np.float32))

    #     return cropped_tensor

    def extract_features(img_path, mask_path):
        img = sitk.ReadImage(img_path[1])
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
            '/app/ingested_program/sample_code_submission/Dataset105_Breast/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres',
            use_folds=(0,), checkpoint_name='checkpoint_final.pth')
        
        # === Build nnUNet-compatible input images folder ===
        nnunet_input_images = os.path.join(output_dir, 'nnunet_input_images')
        os.makedirs(nnunet_input_images, exist_ok=True)

        # === Participants can modify how they prepare input ===
        patient_ids = self.dataset.get_patient_id_list()
        for patient_id in patient_ids:
            images = self.dataset.get_dce_mri_path_list(patient_id)
            # Select the image or images to be used to predict the final segmentation
            # For example, using only the first post-contrast image
            first_post_contrast_image = images[1]
            # Save the image in the nnUNet format (ending in _0000.nii.gz)
            nnunet_image_path = os.path.join(nnunet_input_images, f"{patient_id}_0001_0000.nii.gz")
            # Copy and rename the image to the nnUNet format
            shutil.copy(first_post_contrast_image, nnunet_image_path)

        # === Output folder for raw nnUNet segmentations ===
        output_dir_nnunet = os.path.join(output_dir, 'nnunet_seg')
        os.makedirs(output_dir_nnunet, exist_ok=True)

        # === Call nnUNetv2 prediction ===
        nnunet_images = [[os.path.join(nnunet_input_images, f)] for f in os.listdir(nnunet_input_images)]
        # IMPORTANT: the only method that works inside the Docker container is predict_from_files_sequential
        # This method will predict all images in the list and save them in the output directory
        ret = predictor.predict_from_files_sequential(nnunet_images, output_dir_nnunet, save_probabilities=False,
                                                       overwrite=True, folder_with_segs_from_prev_stage=None)
        print("Predictions saved to:", os.listdir(output_dir_nnunet))
        
        # Folder to store predicted segmentation masks
        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        os.makedirs(output_dir_final, exist_ok=True)

        # === Optional post-processing step ===
        # For example, you can threshold the predictions or apply morphological operations
        # Here, we iterate through the predicted segmentations and apply the breast mask to each segmentation
        # to remove false positives outside the breast region
        for patient_id in self.dataset.get_patient_id_list():
            seg_path = os.path.join(output_dir_nnunet, f"{patient_id}_0001.nii.gz")
            if not os.path.exists(seg_path):
                print(f'{seg_path} NOT FOUND!')
                continue
            
            segmentation = sitk.ReadImage(seg_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            
            patient_info = self.dataset.read_json_file(patient_id)
            if not patient_info or "primary_lesion" not in patient_info:
                continue
            
            coords = patient_info["primary_lesion"]["breast_coordinates"]
            x_min, x_max = coords["x_min"], coords["x_max"]
            y_min, y_max = coords["y_min"], coords["y_max"]
            z_min, z_max = coords["z_min"], coords["z_max"]
            
            masked_segmentation = np.zeros_like(segmentation_array)
            masked_segmentation[x_min:x_max, y_min:y_max, z_min:z_max] = \
                segmentation_array[x_min:x_max, y_min:y_max, z_min:z_max]            
            masked_seg_image = sitk.GetImageFromArray(masked_segmentation)
            masked_seg_image.CopyInformation(segmentation)

            # MANDATORY: the segmentation masks should be named using the patient_id
            final_seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
            sitk.WriteImage(masked_seg_image, final_seg_path)

        # # Save path for Task 2 if needed
        # self.predicted_segmentations = output_dir_final

        predictions = []
    
        model = joblib.load("/app/ingested_program/sample_code_submission/ML/xgb_pcr_model.pkl")
        scaler = joblib.load("/app/ingested_program/sample_code_submission/ML/scaler.pkl")
    
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
            
            features = extract_features(image_paths, seg_path)
            
            # Prepare input as DataFrame
            numeric_values = np.array([[features[col] for col in numeric_cols]])
            failed_flag = np.array([[features["segmentation_failed"]]])

            # Scale numeric features
            scaled_numeric = scaler.transform(numeric_values)
            
            # Combine with segmentation_failed
            input_features = np.hstack([scaled_numeric, failed_flag])
            input_df = pd.DataFrame(input_features, columns=feature_names)

            # Predict
            y_prob = model.predict_proba(input_df)[:, 1][0]
            y_pred = model.predict(input_df)[0]

            # Fallback rule if segmentation failed
            if features["segmentation_failed"] == 1:
                y_prob = 0.5
                y_pred = 0
                
            pcr_prediction = int(y_pred)
            probability = float(y_prob)
            
            predictions.append({
                "patient_id": patient_id,
                "pcr": pcr_prediction,
                "score": probability
            })

        return output_dir_final, pd.DataFrame(predictions)