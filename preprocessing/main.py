import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import *

# Define the dataset paths and folders
dataset_path = '/data/nnUNet/Cropped'
images_folder = dataset_path + '/final_images'
output_folder = os.path.join(dataset_path, 'processed_cropped/images')

# Create output directory if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# Process all patient folders
processed_count = 0

for patient_id in sorted(os.listdir(images_folder)):  
    patient_path = os.path.join(images_folder, patient_id)

    if os.path.isdir(patient_path):
                
        # Compute mean and std across all phases
        mean, std = compute_patient_statistics(images_folder, patient_id)

        if mean is None or std is None:
            continue  # Skip patient if no valid images found

        # Load and normalize only phase 001
        phase_index = 1  # Only process phase 001
        file_name = f"{patient_id}_{phase_index:04d}.nii.gz"
        file_path = os.path.join(images_folder, patient_id, file_name)
        
        output_file_path = os.path.join(output_folder, f"{patient_id}_{phase_index:04d}.nii.gz")

        # Skip if already processed
        if os.path.exists(output_file_path):
            # print(f"✔️ Already processed: {patient_id}")
            continue

        if os.path.exists(file_path):
            image_sitk = sitk.ReadImage(file_path, sitk.sitkFloat32)
            
            # bias_corrected_sitk = bias_correction_sitk(image_sitk, otsu_threshold=True, shrink_factor=4) # newly added

            # denoised_sitk = nlmeans_denoise_sitk(bias_corrected_sitk, patch_size=5, patch_distance=6, h=0.8) # newly added
            
            clipped_sitk = clip_image_sitk(image_sitk, percentiles=[0.1, 99.9]) # newly added
            
            normalized_sitk = zscore_normalization_sitk(clipped_sitk, mean, std) 

            # Resample the normalized image to isotropic resolution
            resampled_sitk = resample_sitk(normalized_sitk, new_spacing=[1,1,1], interpolator=sitk.sitkBSpline)
            print('Original image size:', normalized_sitk.GetSize())
            print('Resampled image size:', resampled_sitk.GetSize())
            

            # Save the resampled image
            sitk.WriteImage(resampled_sitk, output_file_path, useCompression=True)

            print(f"Processed and saved: {output_file_path}")
            processed_count += 1

        else:
            print(f"Warning: {file_path} not found for {patient_id}")

print(f"Successfully processed {processed_count} patients (only phase 001).")