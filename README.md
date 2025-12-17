# MAMA-MIA Challenge 2025 – Final Submission Code

This repository contains the code for our final submission to the **MAMA-MIA Challenge 2025** (MICCAI 2025). The MAMA-MIA challenge focuses on two tasks using a large multi-center breast DCE-MRI dataset [1]:

1. **Primary Tumor Segmentation** in dynamic contrast-enhanced MRI (DCE-MRI) scans  
2. **Treatment Response Prediction**, specifically predicting pathologic complete response (pCR) to neoadjuvant chemotherapy from the MRI data  

### 📄 Paper
Our methodology and findings are detailed in:  
**[Can We Teach AI to Understand Breast Tumour Behaviour? Our MAMA-MIA Challenge Journey](https://link.springer.com/chapter/10.1007/978-3-032-05559-0_25)**

---

## 🖋️ Citation

If you use this code or refer to our work, please cite it as follows:

```bibtex
@InProceedings{Awwad2026MAMAMIA,
author="Awwad, Hadeel and Vilanova, Joan C. and Mart{\'i}, Robert",
title="Can We Teach AI to Understand Breast Tumour Behaviour? Our MAMA-MIA Challenge Journey",
booktitle="Artificial Intelligence and Imaging for Diagnostic and Treatment Challenges in Breast Care",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="248--257",
doi={10.1007/978-3-032-05559-0_25}
}
```

<!-- We provide code for both tasks. The `sample_code_submission` folder includes the files submitted for the Validation phase and Test phase of the challenge. -->

<!-- <img width="3511" height="2174" alt="framework-7" src="https://github.com/user-attachments/assets/fdaada16-79ea-44be-a2dc-0b8af6d53f7a" /> -->

---
<!-- 
## Segmentation (Task 1)

### Preprocessing

We selected three representative DCE-MRI time points as input channels: **Pre-contrast, First Post-contrast, and Second Post-contrast**. These phases are consistently available across all datasets. Preprocessing steps included:

1. **Cropping**: Bounding box cropping of MRI volumes and masks to the affected breast region.
2. **Z-score Normalization**: Percentile-based clipping and normalization using mean and std from the pre-contrast phase.
3. **Resampling**: All volumes resampled to isotropic spacing of **1×1×1 mm**.

The three normalized MRI phases were stacked into a multi-channel 3D volume for model input.

### Model Training

We used **nnU-Net** [2] with the **Residual Encoder Preset L** configuration [5] for segmentation. The training/validation split followed the official MAMA-MIA setup. We disabled nnU-Net's default normalization to retain our customized preprocessing. See the nnU-Net documentation for further details on residual encoder presets.

### Postprocessing

To map the predicted segmentation masks back to the original image space:

1. **Resample to original spacing**
2. **Restore original size** based on recorded cropping coordinates

All preprocessing scripts are available in `preprocessing/main.ipynb`.
All postprocessing scripts are available in `postprocessing/main.py`

---

## Treatment Response Prediction (Task 2)

All relevant functions for classification are located in the `machine_learning/` folder. Patients with unknown pCR labels were assigned to the non-pCR class.

### Preprocessing

Using the segmentation masks from Task 1:

1. Extract tumor regions from all available DCE-MRI phases.
2. Apply percentile clipping and Z-score normalization per image.
3. Stack images into a 4D volume.

### Feature Extraction

We extracted:

- **Temporal dynamics features** from the tumor region in 4D.
- **Shape radiomics** using the PyRadiomics library [3].

A `segmentation_failed` flag marks cases without detected tumors; these are assigned the non-pCR label.

### Classifier Training

We used **XGBoost** [4] with 5-fold stratified cross-validation to handle class imbalance. No explicit feature selection was applied. All extracted features were used.

---

## References

[1] Garrucho, L. et al. (2025). *Advancing Generalizability and Fairness in Breast MRI Tumour Segmentation and Treatment Response Prediction*. Scientific Data. https://doi.org/10.1038/s41597-025-04707-4  

[2] Isensee, F. et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods, 18(2), 203–211. https://doi.org/10.1038/s41592-020-01008-z

[3] van Griethuysen, J. J. M. et al. (2017). *Computational Radiomics System to Decode the Radiographic Phenotype*. Cancer Research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339

[4] Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016. https://doi.org/10.1145/2939672.2939785

[5] Isensee, F. et al. (2024). *nnU-Net revisited: A call for rigorous validation in 3D medical image segmentation*. arXiv preprint arXiv:2403.07288 -->
