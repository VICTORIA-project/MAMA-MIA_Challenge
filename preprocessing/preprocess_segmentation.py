import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import *

# Define the dataset paths and folders
dataset_path = '/data/nnUNet/Cropped'
input_folder = dataset_path + '/final_masks'
output_folder = os.path.join(dataset_path, 'processed_cropped/segmentation')

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process all patient folders
processed_count = 0

# === Process Each File ===
for filename in sorted(os.listdir(input_folder)):
    if not filename.endswith(".nii.gz"):
        continue

    patient_id = filename.replace(".nii.gz", "")
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"{patient_id}.nii.gz")

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"✔️ Already processed: {patient_id}")
        continue

    if os.path.exists(input_path):
        print(f"🔄 Processing: {patient_id}")
        image_sitk = sitk.ReadImage(input_path, sitk.sitkUInt8)
        # Resample the normalized image to isotropic resolution
        resampled_sitk = resample_sitk(image_sitk, new_spacing=[1,1,1], interpolator=sitk.sitkNearestNeighbor)
        print('Original image size:', image_sitk.GetSize())
        print('Resampled image size:', resampled_sitk.GetSize())
        
        output_file_path = os.path.join(output_folder, f"{patient_id}.nii.gz")

        # Save the resampled image
        sitk.WriteImage(resampled_sitk, output_file_path, useCompression=True)

        print(f"Processed and saved: {output_file_path}")
        processed_count += 1

    else:
        print(f"Warning: {file_path} not found for {patient_id}")

print(f"Successfully processed {processed_count} patients (segmentation mask).")