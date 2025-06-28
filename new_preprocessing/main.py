import os
import json
import glob
import SimpleITK as sitk


# def extract_breast_coordinates(json_folder):

#     coords_dict = {}

#     for filename in os.listdir(json_folder):
#         if not filename.endswith(".json"):
#             continue

#         json_path = os.path.join(json_folder, filename)
#         patient_id = os.path.splitext(filename)[0]  # remove ".json"

#         try:
#             with open(json_path, "r") as f:
#                 data = json.load(f)

#             if data.get("patient_id") != patient_id:
#                 raise ValueError(f"Patient ID mismatch in file {filename}")

#             if "primary_lesion" not in data or "breast_coordinates" not in data["primary_lesion"]:
#                 raise ValueError(f"Missing breast coordinates for {patient_id}")

#             coords = data["primary_lesion"]["breast_coordinates"]
            
#             coords_dict[patient_id] = coords
#             # x_min, x_max = coords["x_min"], coords["x_max"]
#             # y_min, y_max = coords["y_min"], coords["y_max"]
#             # z_min, z_max = coords["z_min"], coords["z_max"]

#             # roi_start = [x_min, y_min, z_min]
#             # roi_size = [x_max - x_min, y_max - y_min, z_max - z_min]

#         except Exception as e:
#             print(f"❌ Error processing {filename}: {e}")

#     return coords_dict

# json_folder = "/data/MAMA-MIA/patient_info_files"
# breast_coords = extract_breast_coordinates(json_folder)

# coords = breast_coords.get("DUKE_002")

# if coords is not None:
#     x_min, x_max = coords["x_min"], coords["x_max"]
#     y_min, y_max = coords["y_min"], coords["y_max"]
#     z_min, z_max = coords["z_min"], coords["z_max"]
# else:
#     print(f"No coordinates found for {patient_id}")
 
##################################################################################3

# def reverse_resample_mask(resampled_mask, original_image):
#     """
#     Resample a segmentation mask back to the original image's geometry.

#     Parameters:
#     - resampled_mask: sitk.Image (e.g., model output mask)
#     - original_image: sitk.Image (original image before resampling)

#     Returns:
#     - reverted_mask: sitk.Image (mask back in original spacing and grid)
#     """
#     resample = sitk.ResampleImageFilter()
#     resample.SetReferenceImage(original_image)
#     resample.SetInterpolator(sitk.sitkNearestNeighbor)
#     resample.SetTransform(sitk.Transform())  # Identity transform
#     resample.SetOutputSpacing(original_image.GetSpacing())
#     resample.SetSize(original_image.GetSize())
#     resample.SetOutputOrigin(original_image.GetOrigin())
#     resample.SetOutputDirection(original_image.GetDirection())
    
#     return resample.Execute(resampled_mask)

# # === Set your file paths ===
# resampled_mask_path = "/data/nnUNet/Cropped/segmentation/{patient_id}.nii.gz"     # Mask from model
# original_image_path = "/data/MAMA-MIA/images/{patient_id}/{patient_id}.nii.gz"     # Original image with correct spacing
# output_mask_path = resampled_mask_path  # Overwrite or new file

# # === Load images ===
# resampled_mask = sitk.ReadImage(resampled_mask_path, sitk.sitkUInt8)  # For masks, use UInt8
# original_image = sitk.ReadImage(original_image_path, sitk.sitkFloat32)

# # === Reverse resample ===
# reverted_mask = reverse_resample_mask(resampled_mask, original_image)

# # === Save the reverted mask ===
# sitk.WriteImage(reverted_mask, output_mask_path)

# print(f"✅ Reverted mask saved to: {output_mask_path}")

#############################################################

# import os
# import glob
# import json
# import SimpleITK as sitk

# def crop_all_images_and_masks(
#     images_root,
#     masks_root,
#     json_folder,
#     output_image_root,
#     output_mask_root
# ):
#     os.makedirs(output_image_root, exist_ok=True)
#     os.makedirs(output_mask_root, exist_ok=True)

#     patient_folders = sorted(glob.glob(os.path.join(images_root, "*")))

#     for folder_path in patient_folders:
#         patient_id = os.path.basename(folder_path).strip()
#         json_path = os.path.join(json_folder, f"{patient_id}.json")

#         if not os.path.exists(json_path):
#             # print(f"❌ JSON not found for {patient_id}")
#             continue

#         # === Load breast_coordinates ===
#         try:
#             with open(json_path, "r") as f:
#                 data = json.load(f)

#             coords = data["primary_lesion"]["breast_coordinates"]
#             x_min, x_max = coords["x_min"], coords["x_max"]
#             y_min, y_max = coords["y_min"], coords["y_max"]
#             z_min, z_max = coords["z_min"], coords["z_max"]
#         except Exception as e:
#             print(f"❌ Failed to read coordinates for {patient_id}: {e}")
#             continue

#         # === Load and crop the mask once ===
#         mask_path = os.path.join(masks_root, f"{patient_id}.nii.gz")
#         if not os.path.exists(mask_path):
#             print(f"❌ Mask not found for {patient_id}")
#             continue

#         try:
#             mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)
#             mask_np = sitk.GetArrayFromImage(mask_sitk)
#             cropped_mask_np = mask_np[x_min:x_max, y_min:y_max, z_min:z_max]
#             cropped_mask_sitk = sitk.GetImageFromArray(cropped_mask_np)

#             # Get metadata from mask (same spacing/direction as images)
#             spacing = mask_sitk.GetSpacing()
#             direction = mask_sitk.GetDirection()
#             origin = mask_sitk.TransformIndexToPhysicalPoint((z_min, y_min, x_min))

#             cropped_mask_sitk.SetSpacing(spacing)
#             cropped_mask_sitk.SetDirection(direction)
#             cropped_mask_sitk.SetOrigin(origin)

#             # Save cropped mask
#             out_mask_path = os.path.join(output_mask_root, f"{patient_id}.nii.gz")
#             os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
#             sitk.WriteImage(cropped_mask_sitk, out_mask_path, useCompression=True)
#             print(f"✅ Cropped mask saved for {patient_id}")

#         except Exception as e:
#             print(f"❌ Failed to process mask for {patient_id}: {e}")
#             continue

#         # === Now process each image in the folder ===
#         image_files = sorted(glob.glob(os.path.join(folder_path, "*.nii.gz")))
#         if not image_files:
#             print(f"❌ No images found for {patient_id}")
#             continue

        # for image_path in image_files:
        #     image_filename = os.path.basename(image_path)
        #     image_name = os.path.splitext(image_filename)[0]

        #     try:
        #         img_sitk = sitk.ReadImage(image_path, sitk.sitkFloat32)
        #         img_np = sitk.GetArrayFromImage(img_sitk)

        #         shape = img_np.shape  # (Z, Y, X) —> but coordinates map as: x→axis0, y→axis1, z→axis2

        #         # Validate bounds
        #         if not (0 <= x_min < x_max <= shape[0] and
        #                 0 <= y_min < y_max <= shape[1] and
        #                 0 <= z_min < z_max <= shape[2]):
        #             print(f"❌ Coordinates out of bounds for {patient_id} / {image_name}")
        #             continue

        #         # Crop image
        #         cropped_img_np = img_np[x_min:x_max, y_min:y_max, z_min:z_max]
        #         cropped_img_sitk = sitk.GetImageFromArray(cropped_img_np)

        #         # Apply same metadata as original image
        #         spacing = img_sitk.GetSpacing()
        #         direction = img_sitk.GetDirection()
        #         origin = img_sitk.TransformIndexToPhysicalPoint((z_min, y_min, x_min))

        #         cropped_img_sitk.SetSpacing(spacing)
        #         cropped_img_sitk.SetDirection(direction)
        #         cropped_img_sitk.SetOrigin(origin)

        #         # Save image
        #         patient_output_folder = os.path.join(output_image_root, patient_id)
        #         os.makedirs(patient_output_folder, exist_ok=True)

#                 out_img_path = os.path.join(patient_output_folder, f"{image_name}.nii.gz")
#                 sitk.WriteImage(cropped_img_sitk, out_img_path, useCompression=True)

#                 print(f"✅ Cropped and saved: {patient_id} / {image_name}")

#             except Exception as e:
#                 print(f"❌ Failed for {patient_id} / {image_name}: {e}")

# # === Run batch cropping ===
# crop_all_images_and_masks(
#     images_root="/data/nnUNet/Cropped/images",
#     masks_root="/data/nnUNet/Cropped/segmentation",
#     json_folder="/data/MAMA-MIA/patient_info_files",
#     output_image_root="/data/nnUNet/Cropped/final_images",
#     output_mask_root="/data/nnUNet/Cropped/final_masks"
# )
########################################################################

import numpy as np
import SimpleITK as sitk

import os
import json
import numpy as np
import SimpleITK as sitk

def restore_mask_to_original_space_from_paths(
    patient_id,
    cropped_mask_path,
    original_image_path,
    json_folder,
    output_path
):
    """
    Restores a cropped mask to its original spatial size using metadata and VOI coordinates from a JSON file.

    Args:
        patient_id (str): Patient identifier (e.g., 'DUKE_019')
        cropped_mask_path (str): Path to the cropped mask (.nii.gz)
        original_image_path (str): Path to the full-size original image (or mask) for shape and metadata
        json_folder (str): Folder containing JSON files named like {patient_id}.json
        output_path (str): Path to save the restored mask
    """
    # === Load crop coordinates from JSON ===
    json_path = os.path.join(json_folder, f"{patient_id}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    try:
        coords = data["primary_lesion"]["breast_coordinates"]
        x_min, x_max = coords["x_min"], coords["x_max"]
        y_min, y_max = coords["y_min"], coords["y_max"]
        z_min, z_max = coords["z_min"], coords["z_max"]
    except KeyError:
        raise ValueError(f"Missing 'breast_coordinates' in {json_path}")

    crop_start = (x_min, y_min, z_min)

    # === Load cropped mask ===
    cropped_mask_sitk = sitk.ReadImage(cropped_mask_path, sitk.sitkUInt8)
    cropped_mask_np = sitk.GetArrayFromImage(cropped_mask_sitk)

    # === Load original image to get shape and metadata ===
    original_sitk = sitk.ReadImage(original_image_path)
    original_shape = sitk.GetArrayFromImage(original_sitk).shape  # (Z, Y, X)

    # === Prepare blank full-size array and insert cropped mask ===
    restored_np = np.zeros(original_shape, dtype=cropped_mask_np.dtype)
    x_min, y_min, z_min = crop_start
    x_max = x_min + cropped_mask_np.shape[0]
    y_max = y_min + cropped_mask_np.shape[1]
    z_max = z_min + cropped_mask_np.shape[2]

    restored_np[x_min:x_max, y_min:y_max, z_min:z_max] = cropped_mask_np

    # === Convert back to SimpleITK and restore metadata ===
    restored_sitk = sitk.GetImageFromArray(restored_np)
    restored_sitk.SetSpacing(original_sitk.GetSpacing())
    restored_sitk.SetDirection(original_sitk.GetDirection())
    restored_sitk.SetOrigin(original_sitk.GetOrigin())

    # === Save the result ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(restored_sitk, output_path)

    print(f"✅ Restored mask saved to: {output_path}")


restore_mask_to_original_space_from_paths(
    patient_id="DUKE_019",
    cropped_mask_path="/data/nnUNet/Cropped/final_masks/DUKE_019.nii.gz",
    original_image_path="/data/nnUNet/Cropped/images/DUKE_019/DUKE_019_0000.nii.gz",
    json_folder="/data/MAMA-MIA/patient_info_files",
    output_path="/data/nnUNet/RestoredMasks/DUKE_019/DUKE_019_0000_restored.nii.gz"
)