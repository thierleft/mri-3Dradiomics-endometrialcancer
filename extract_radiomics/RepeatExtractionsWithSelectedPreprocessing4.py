# -*- coding: utf-8 -*-
"""
Repeat radiomics features extractions in segmented VOIs knowing
which subject is category 0 (e.g. early FIGO stage) or 1 (e.g. advanced FIGO stage).
Perform on training and testing datasets!

Not for clinical use.
SPDX-FileCopyrightText: 2021 Medical Physics Unit, McGill University, Montreal, CAN
SPDX-FileCopyrightText: 2021 Thierry Lefebvre
SPDX-FileCopyrightText: 2021 Peter Savadjiev
SPDX-License-Identifier: MIT
"""


from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import numpy as np
from os import listdir
import os as os
from os.path import join
import pandas as pd # This module is used to save radiomics features to CSV files

## INPUT == PATIENT DATA TO DICHOTOMIZE BENIGN vs. MALIGNANT AND PREPROCESSING STEPS MAX ICC resampledPixelSpacing binWidth
## OUTPUT == RAD FTS IN CSV FILES (TO BE REPEATED FOR TRAINING AND TESTING)

modality = 'MRI_SEQUENCE' # insert name of MRI sequence of interest (e.g. DCE2, ADC, DWI, etc.)

### ACCESS IMAGES AND SEGMENTATIONS ###

# Define path to imgs in .nii format
mypathimg = 'MYPROJECTFILEPATH/IMG/'+modality+'/' #repeat for training and testing

# Define path to segmentations (will need to be changed depending for each set of segmentations)
mypathseg = 'MYPROJECTFILEPATH/SEG/'+modality+'/' #repeat for training and testing

# List files in imgs directory
dirsimg = listdir(mypathimg)
dirsimg.sort()

# List files in segmentations directory
dirsseg = listdir(mypathseg)
dirsseg.sort()

# Instentiate variables for img and segmentation paths
fullpathsimg = []
fullpathsseg = []

# Iterate on imgs and segmentations to list full paths to imgs
for dir1 in dirsimg:
    fullpathsimg.append(join(mypathimg,dir1))
for dir2 in dirsseg:
    fullpathsseg.append(join(mypathseg,dir2))


# Define paths to save extracted radiomics features to .csv files
mypathsave1 = 'MYPROJECTFILEPATH/FINAL_EXTRACTIONS/1/'+modality+'/' # e.g. where 1 = advanced FIGO stage
mypathsave0 = 'MYPROJECTFILEPATH/FINAL_EXTRACTIONS/0/'+modality+'/' # 0 = early FIGO stage

# Verify if file exists, and create it if not
if not os.path.isdir(mypathsave1):
    os.makedirs(mypathsave1)
if not os.path.isdir(mypathsave0):
    os.makedirs(mypathsave0)

# Define selected preprocessing parameters
params = {}
params['normalize']         =    True # Selected normalization
params['removeOutliers']    =    3 # IBSI recommandations (n*sig where n = 3 * sig = std dev)
params['resampledPixelSpacing']= [1, 1, 1] # Selected resampling to isotropic voxel size      
params['interpolator']      =    3 # sitkBspline interpolator for isotropic voxel resampling
params['binWidth']          =    30 # Selected bin width

# Instantiate the radiomics features extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# File with patient data to be used to split sample in two groups for classification
xlsxFile = pd.read_excel('MYPROJECTFILEPATH/MYEXCELFILE.xlsx')

# Loop over all patients'images
for i in range(len(dirsseg)):
    print(dirsseg[i][0:-4])
    print(i)
            
    # Extract radiomics features
    resultsRadFts = extractor.execute(fullpathsimg[i],fullpathsseg[i])

    # Preprend modality string prefix to feature names
    outRadFts.index = modality + '_' + outRadFts.index

    # Format features set to Pandas Data Frame
    outRadFts = pd.DataFrame([resultsRadFts]).T
    
    # Verification string corresponding to patient name in the img file name
    verifStr = dirsimg[i][0:-4]
    
    # Here MRN is the column with all patient IDs corresponding img file names
    if np.any(xlsxFile.isin({'MRN': [verifStr]})): # Verify if patient is in the excel file
        idx = np.where(xlsxFile.isin({'MRN': [verifStr]}))[0][0]
        listValues = xlsxFile.iloc[idx,:] # list of columns associated with the current patient
        if np.any(listValues.isin(['Benign'])): # save in 0 if Benign in the excel file
            outRadFts.to_csv((join(mypathsave0,dirsseg[i][0:-4])+'.csv'))
        elif np.any(listValues.isin(['Malignant'])): # save if Malignant in the excel file
            outRadFts.to_csv((join(mypathsave1,dirsseg[i][0:-4])+'.csv'))
