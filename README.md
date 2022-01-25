# mri-3Dradiomics-endometrialcancer

This repository contains the radiomics and machine learning code associated with the study by Lefebvre et al. published in Radiology (Lefebvre et al. Development and Validation of Multiparametric MRI-based Radiomics Models for the Risk Stratification of Endometrial Cancer. Radiology. 2022).  

This work is largerly based on contributions and Python packages developed by others and reported previously, mainly:
* `pyradiomics` (van Griethuysen et al. Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. 2017)
* `scikit-learn` (Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830. 2011)

## Pyradiomics pipeline

STEPS 1 TO 3 ARE ONLY DONE IN THE TRAINING DATASET

STEPS 4 AND 5 HAVE TO BE REPEATED TWICE, ONCE ON THE TRAINING AND ONCE ON THE TESTING DATASET


1. `PreprocessingRadiomicsExtractions1.py`

Define combinations of preprocessing steps for radiomics features extraction. Extractions have to be performed on .nii images and segmentations. This script has to be repeated for each set of segmentations to evaluate ICCs.


2. `EvaluateICCFromPreprocessedExtractions2.py`


Evaluate ICC for each radiomics feature by comparing extractions from each set of segmentations (e.g. normal, eroded, dilated segmentations).


3. `AnalyzeICCsAndGenerateListofExcludedFeatures3.m`

Analyze ICC for each radiomics feature and each preprocessing parameters set. Select most stable set of preprocessing parameters and features to be excluded from further modeling.


4. `RepeatExtractionsWithSelectedPreprocessing4.py`


Repeat radiomics features extractions in segmented VOIs knowing which subject is category 0 (e.g. benign) or 1 (e.g. malignant). Perform on training and testing datasets!


5. `FormatRadiomicsFeaturesForAnalysis5.py`

Format Radiomics Features for Analysis.
Save features in separate files according to classification categories (e.g. benign vs. malignant) and excluding unreproducible features prior to further analyses.



