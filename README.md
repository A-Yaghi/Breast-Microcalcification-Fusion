# Feature-Fusion Techniques for Breast Microcalcification Classification

This repository contains the architectural source code and experimental pipelines for evaluating multimodal feature-fusion strategies in breast cancer classification. The project investigates how to optimally combine high-dimensional medical imaging with lower-dimensional structured tabular data (clinical demographics and explicitly engineered radiomics).

## Project Overview

Breast microcalcifications (MCs) are critical early radiological biomarkers for malignancy. While clinical diagnosis inherently combines visual findings with patient context, deep learning models often suffer from modality dominance when fusing these data types. This research evaluates two distinct multimodal integration strategies:
*   **Decision-Level Late Fusion:** Processes modalities independently and aggregates final predictions via a Logistic Regression meta-classifier.
*   **Feature-Level Intermediate Fusion (DAFT):** Utilizes the Dynamic Affine Feature Map Transform to dynamically scale and shift intermediate convolutional feature maps based on tabular inputs.

These architectures are evaluated across two distinct clinical environments:
1.  **In-Vivo Clinical Screening:** Combining 2D mammography regions of interest (ROIs) from the open-source EMBED dataset with demographic tabular data.
2.  **Ex-Vivo Pre-Clinical Research:** Fusing a private dataset of high-resolution 3D Micro-CT volumetric patches of individual MCs with 979 handcrafted PyRadiomics features.

## Key Findings

*   **2D Mammography Cohort:** The intermediate DAFT approach effectively mitigated the poor performance of unimodal baselines in a data-constrained setting, achieving a 5-fold cross-validation mean AUC of 0.6750.
*   **3D Micro-CT Cohort:** Mathematically engineered radiomic features provided a highly competitive predictive baseline for individual 3D MCs. The tabular-only XGBoost model achieved an AUC of 0.8397, outperforming the pure 3D CNN baseline (AUC 0.6468).
*   **Ablation Studies:** Restricting the tabular input to the top 200 radiomic features via recursive feature elimination improved the DAFT architecture's AUC to 0.8328. Furthermore, applying a convex combination constraint (Softmax) to the Late Fusion meta-learner revealed that the optimizer assigns >99% of the voting weight to the radiomic modality.
*   **Statistical Significance:** Patient-clustered bootstrapping (10,000 iterations) and McNemar's Test revealed no statistically significant differences between the multimodal fusion networks and the pure radiomic XGBoost baseline, suggesting a high degree of information redundancy between the spatial convolutional features and the engineered radiomic descriptors in this specific ex-vivo environment.

## Repository Structure

*   `docs/`: Contains the full master thesis manuscript detailing the theoretical background, mathematical formulations, and complete ablation studies.
*   `src/3D/preprocessing/`: Logic for 3D aspect-ratio-preserving volumetric normalization. *(Note: The 3D Micro-CT dataset is proprietary and excluded from this repository).*
*   `src/2D/preprocessing/`: Logic for 2D spatial fixed-window cropping.
*   `src/3D/models/`: tabular unimodels (XGBOOST, SVM), PyTorch implementations of the unimodal vision baselines (CNN, ResNet-18), the DAFT bottleneck module, and the Late Fusion meta-classifier for the 3D part.
*   `src/2D/models/`: tabular unimodels (XGBOOST, SVM), PyTorch implementations of the unimodal vision baselines (CNN, ResNet-18), the DAFT bottleneck module, and the Late Fusion meta-classifier for the 2D part.
*   `src/3D/Statistics/`: Scripts for patient-level cross-validation splitting and statistical significance bootstrapping.

## Technologies Used
*   Python
*   PyTorch (Deep Learning & DAFT implementation)
*   XGBoost & Scikit-learn (Tabular baselines & Meta-classifiers)
