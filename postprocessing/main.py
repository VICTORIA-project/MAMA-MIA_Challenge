def resample_to_spacing(image_sitk, target_spacing):
    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image_sitk.GetDirection())
    resample.SetOutputOrigin(image_sitk.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(image_sitk)

def restore_mask_to_original_space_from_data(patient_id, predicted_mask_path, original_image_path, coords, output_path):
    original_sitk = sitk.ReadImage(original_image_path)
    restored_sitk = sitk.Image(original_sitk.GetSize(), sitk.sitkUInt8)
    restored_sitk.CopyInformation(original_sitk)

    cropped_mask_sitk = sitk.ReadImage(predicted_mask_path, sitk.sitkUInt8)
    resampled_pred = resample_to_spacing(cropped_mask_sitk, coords["original_spacing"])
    origin_phys = coords["origin_phys"]
    resampled_pred.SetOrigin(origin_phys)

    destination_index = original_sitk.TransformPhysicalPointToIndex(origin_phys)

    restored_sitk = sitk.Paste(
        destinationImage=restored_sitk,
        sourceImage=resampled_pred,
        sourceSize=resampled_pred.GetSize(),
        sourceIndex=[0, 0, 0],
        destinationIndex=destination_index
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(restored_sitk, output_path, useCompression=True)

def restore_all_masks(cropping_metadata_path, predicted_masks_dir, original_images_root, output_root):
    with open(cropping_metadata_path) as f:
        metadata = json.load(f)

    for patient_id, coords in tqdm(metadata.items(), desc="Restoring masks"):
        predicted_mask_path = os.path.join(predicted_masks_dir, f"{patient_id}.nii.gz")
        original_image_path = os.path.join(original_images_root, patient_id, f"{patient_id}_0001.nii.gz")
        output_path = os.path.join(output_root, patient_id, f"{patient_id}_restored.nii.gz")

        if not os.path.exists(predicted_mask_path):
            print(f"⚠️ Missing prediction for {patient_id}")
            continue
        if not os.path.exists(original_image_path):
            print(f"⚠️ Missing original image for {patient_id}")
            continue

        restore_mask_to_original_space_from_data(
            patient_id=patient_id,
            predicted_mask_path=predicted_mask_path,
            original_image_path=original_image_path,
            coords=coords,
            output_path=output_path
        )
        
restore_all_masks(
    cropping_metadata_path="/path/to/cropping_coords.json",
    predicted_masks_dir="/results/nnUNet/nnUNet_results/Dataset108_multiC/nnUNetTrainer__.../fold_0/validation",
    original_images_root="/data/MAMA-MIA/images",
    output_root="/data/nnUNet/RestoredMasks"
)