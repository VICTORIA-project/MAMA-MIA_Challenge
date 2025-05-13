import pandas as pd
import json
import os

# Load the CSV file
csv_path = '/data/One_dataset/train_test_splits.csv'
df = pd.read_csv(csv_path)

# Define the base paths
images_path = "/data/One_dataset/images/"
label_path = "/data/One_dataset/segmentation/"

# Initialize the structure for the JSON
json_data = {
    "training": [],
    "validation": []
}

def find_image_path(case_id):
    folder = os.path.join(images_path, case_id)
    # List .nii.gz files inside the case folder
    nii_files = [f for f in os.listdir(folder) if f.endswith(".nii.gz")]
    if len(nii_files) == 0:
        raise FileNotFoundError(f"No NIfTI file found for {case_id} in {folder}")
    elif len(nii_files) > 1:
        print(f"Warning: Multiple NIfTI files found for {case_id}, using the first one")
    return [os.path.join(folder, nii_files[0])]

# Training entries
for case_id in df["train_split"].dropna():
    entry = {
        "image": find_image_path(case_id),
        "label": os.path.join(label_path, f"{case_id}.nii.gz")
    }
    json_data["training"].append(entry)

# Validation entries
for case_id in df["test_split"].dropna():
    entry = {
        "image": find_image_path(case_id),
        "label": os.path.join(label_path, f"{case_id}.nii.gz")
    }
    json_data["validation"].append(entry)

# Save the JSON to a file
output_json_path = "/home/hadeel/MAMA-MIA_Challenge/research-contributions/SwinUNETR/mama-mia/jsons/data_split.json"
with open(output_json_path, "w") as f:
    json.dump(json_data, f, indent=4)

output_json_path