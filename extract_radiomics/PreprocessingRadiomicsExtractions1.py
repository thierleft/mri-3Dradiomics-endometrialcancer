# -*- coding: utf-8 -*-
"""
Define combinations of preprocessing steps for radiomics features extraction.
Extractions have to be performed on .nii images and segmentations.
This script has to be repeated for each set of segmentations to evaluate ICCs.

Not for clinical use.
SPDX-FileCopyrightText: 2021 Medical Physics Unit, McGill University, Montreal, CAN
SPDX-FileCopyrightText: 2021 Thierry Lefebvre
SPDX-FileCopyrightText: 2021 Peter Savadjiev
SPDX-License-Identifier: MIT
"""

# Import required packages
from radiomics import featureextractor # This module is used for interaction with pyradiomics
import numpy as np
from os import listdir
import os as os
import os.path
from os.path import join
import pandas as pd # This module is used to save radiomics features to CSV files

# INPUTS = PATHS FOR IMGS, SEG AND FOR EXPORTATIONS
# INPUTS = PREPROCESSING PARAMETERS binwidthList isoList normzList
# OUTPUTS = CSV CONTAINING RAD FTS FOR EACH PREPROCESSING COMBINATION

modality = 'MRI_SEQUENCE' # insert name of MRI sequence of interest (e.g. DCE2, ADC, DWI, etc.)
myseg =    'SEG'          # insert name of segmentation of interest (SEG, DILSEG, EROSEG)

### ACCESS IMAGES AND SEGMENTATIONS ###

# Define path to imgs in .nii format
mypathimg = 'MYPROJECTFILEPATH/IMG/'+modality+'/'

# Define path to segmentations (will need to be changed depending for each set of segmentations [regular, eroded, and dilated])
mypathseg = 'MYPROJECTFILEPATH/'+myseg+'/'+modality+'/'

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

### IMG PREPROCESSING PARAMETERS PRIOR TO EXTRACTIONS ###

# Define fixated bin width sizes to be tested
binwidthList = np.array([15,20,25,30],dtype = int) 

# Define isotropic voxel sizes (mm^3) to be tested
isoList = np.array([0.5,int(1),int(2),int(3)])

# Normalize images or not? should be considered for MRI data
normzList = np.array([True])


### START RADIOMICS FEATURES EXTRACTIONS ###

for normz in normzList:
    for iso in isoList:
        for binwidth in binwidthList:
            
            params = {}
            params['normalize'] = normz            
            if normz == True:
                params['normalizeScale'] = 300 # scale for MRI data used in literature
            params['removeOutliers'] = 3 # IBSI recommandations (n*sig where n = 3 * sig = std dev)
            params['resampledPixelSpacing'] = [iso, iso, iso]            
            params['interpolator'] = 3 # sitkBspline interpolator for isotropic voxel resampling
            params['binWidth'] = binwidth
            
            isoO=iso
            
            # Instentiate Pyradiomics feature extractor with set of preprocessing parameters
            extractor = featureextractor.RadiomicsFeatureExtractor(**params)
            
            for i in range(len(dirsseg)):
                # Display file name, iteration number, and extraction parameters
                print(dirsseg[i][0:-4])
                print(i)
                print(['Norm? = '+str(normz)+' ; Voxel size = '+str(isoO)+' ; Bin size = '+str(binwidth)])

                # Extract radiomics features
                resultsRadFts = extractor.execute(fullpathsimg[i],fullpathsseg[i])
                
                # Format features set to Pandas Data Frame
                outRadFts = pd.DataFrame([resultsRadFts]).T
                
                if iso == 0.5: # The dot "." in "0.5" while saving to file is problematic
                    isoO = int(5)
                else:
                    isoO = iso
                
                # Save path with informative name (for each set of SEG, EROSEG, DILSEG)
                mypathsave = 'MYPROJECTFILEPATH/PREPROCESSED_EXTRACTIONS/'+modality+'/'+myseg+'/RadFts'+str(normz)+str(isoO)+str(binwidth)
                
                # Verify if file exists, and create it if not
                if not os.path.isdir(mypathsave):
                    os.makedirs(mypathsave)

                # Preprend modality string prefix to feature names
                outRadFts.index = modality + '_' + outRadFts.index

                # Save CSV file of radiomics features with current set of preprocessing parameters
                outRadFts.to_csv((join(mypathsave,dirsseg[i][0:-4])+'.csv'))
  
            
