# -*- coding: utf-8 -*-
"""
Evaluate intra-class correlation coefficients (ICC) for each radiomics feature by comparing extractions from
each set of segmentations (e.g. normal, eroded, dilated segmentations).

Not for clinical use.
SPDX-FileCopyrightText: 2021 Medical Physics Unit, McGill University, Montreal, CAN
SPDX-FileCopyrightText: 2021 Thierry Lefebvre
SPDX-FileCopyrightText: 2021 Peter Savadjiev
SPDX-License-Identifier: MIT
"""

import os 
from os.path import join
import pandas as pd # This module is used to read CSV files of radiomics features
import numpy as np
import pingouin as pg # This statistical module is used to assess ICCs
import scipy.io as sio # This module is used to save ICC as .mat files

# INPUTS = PATHS FOR RAD FTS EXPORTATED FOR EACH SEGM AND EACH PREPROCESSING COMBINATION
# OUTPUTS = ICC FOR EACH PREPROCESSING COMBINATION

modality = 'MRI_SEQUENCE' # insert name of MRI sequence of interest (e.g. DCE2, ADC, DWI, etc.)

basepath = 'MYPROJECTFILEPATH/'

# Paths to each segmentations directory 
mypath =    'MYPROJECTFILEPATH/PREPROCESSED_EXTRACTIONS/'+modality+'/SEG/'
mypathERO = 'MYPROJECTFILEPATH/PREPROCESSED_EXTRACTIONS/'+modality+'/EROSEG/'
mypathDIL = 'MYPROJECTFILEPATH/PREPROCESSED_EXTRACTIONS/'+modality+'/DILSEG/'

mypathsave = 'MYPROJECTFILEPATH/SAVE/'+modality

# Verify if file exists, and create it if not
if not os.path.isdir(mypathsave):
    os.makedirs(mypathsave)

# List files in segmentation directories with files containing radiomics features
# of all patients for each set of preprocessing parameters
pathlabels = os.listdir(mypath)
pathEROlabels = os.listdir(mypathERO)
pathDILlabels = os.listdir(mypathDIL)


# Loop over sub-folders with each set of preprocessing parameters
for label in pathlabels:
    dir1 = join(mypath, label)
    listfiles = os.listdir(dir1)
    
    dir2 = join(mypathERO, label)
    listfiles2 = os.listdir(dir2)
    
    dir3 = join(mypathDIL, label)
    listfiles3 = os.listdir(dir3)

    # get the current label
    current_label = label
    print("Current Label = ",current_label)
    i, j = 0, 0
    
    # Loop over data from all patients in sub-sub-folder
    for file in listfiles:
        # Make sure this patient's features were extracted for each segmentation
        if file in listfiles2 and file in listfiles3:
            
            # Read CSV file of radiomics features for this patient
            global_feature = pd.read_csv(join(dir1,file), delimiter=',')
            
            # Extract radiomics features (37 is the first element which is a radiomics feature for Pyradiomics extractions)
            global_values = global_feature.iloc[37:-1,1]
            global_values = global_values.astype(float)
            global_values = np.array(global_values.values)
            
            # Extract names of radiomics features e.g. original_gldm_DependenceEntropy
            global_names = global_feature.iloc[37:-1,0]
            global_names = np.array(global_names.values)
            
            # Get all radiomics features with header for this patient
            global_feature = np.vstack((global_names, global_values))
            global_feature = pd.DataFrame(global_feature)
            new_hd = global_feature.iloc[0]
            new_hd.iloc[:] = new_hd.iloc[:].astype(str)
            global_feature = global_feature[1:]
            global_feature.columns = new_hd
            global_feature.reset_index(drop = True)
            rescaled_feature = global_feature # This is the final Data Frame we use for the normal segmentations
            
            # Repeat for eroded segmentations
            global_featureERO = pd.read_csv(join(dir2,file), delimiter=',')
            global_valuesERO = global_featureERO.iloc[37:-1,1]
            global_valuesERO  = global_valuesERO.astype(float)
            global_valuesERO  = np.array(global_valuesERO.values)
            global_namesERO  = global_featureERO.iloc[37:-1,0]
            global_namesERO  = np.array(global_namesERO.values)
            global_featureERO  = np.vstack((global_namesERO, global_valuesERO))
            global_featureERO  = pd.DataFrame(global_featureERO)
            new_hdERO = global_featureERO.iloc[0]
            new_hdERO.iloc[:] = new_hdERO.iloc[:].astype(str)
            
            global_featureERO  = global_featureERO[1:]
            global_featureERO.columns = new_hdERO
            global_featureERO.reset_index(drop = True)
            rescaled_featureERO  = global_featureERO # This is the final Data Frame we use for eroded segmentations
            
            # Repeat for dilated segmentations            
            global_featureDIL= pd.read_csv(join(dir3,file), delimiter=',')
            global_valuesDIL= global_featureDIL.iloc[37:-1,1]
            global_valuesDIL= global_valuesDIL.astype(float)
            global_valuesDIL= np.array(global_valuesDIL.values)
            global_namesDIL= global_featureDIL.iloc[37:-1,0]
            global_namesDIL= np.array(global_namesDIL.values)
            global_featureDIL= np.vstack((global_namesDIL, global_valuesDIL))
            global_featureDIL= pd.DataFrame(global_featureDIL)
            new_hdDIL= global_featureDIL.iloc[0]
            new_hdDIL.iloc[:] = new_hdDIL.iloc[:].astype(str)
            
            global_featureDIL= global_featureDIL[1:]
            global_featureDIL.columns = new_hdDIL
            global_featureDIL.reset_index(drop = True)
            rescaled_featureDIL= global_featureDIL # This is the final Data Frame we use for dilated segmentations

            if i == 0 and j == 0:
                global_features = rescaled_feature
                global_featuresERO = rescaled_featureERO
                global_featuresDIL= rescaled_featureDIL
            else:
                # Append radiomics features of each patient into a final Data Frame
                global_features = global_features.append(rescaled_feature,ignore_index=True)
                global_featuresERO = global_featuresERO.append(rescaled_featureERO,ignore_index=True)
                global_featuresDIL= global_featuresDIL.append(rescaled_featureDIL,ignore_index=True)
            
            i+=1

    ii = 0
    
    # Instentiate list of ICC to be stored and saved
    iccList = []
    
    # For each radiomics feature, we assess the ICC across segmentations and patients
    for feats in new_hd:
        
        # Concatenate the current radiomics feature analyzed for all segmentations to evaluate an ICC
        feat_compare = pd.concat([global_features.iloc[:,ii], global_featuresERO.iloc[:,ii],global_featuresDIL.iloc[:,ii]], axis=0)
        
        # Reformat data to be analyzed with Pingouin intraclass_corr function
        rater1 = pd.DataFrame([1]*len(global_features.iloc[:,ii]),columns=['rater'])
        rater2 = pd.DataFrame([2]*len(global_featuresERO.iloc[:,ii]),columns=['rater'])
        rater3 = pd.DataFrame([3]*len(global_featuresDIL.iloc[:,ii]),columns=['rater'])
        raters = pd.concat([rater1,rater2,rater3],axis=0)
        raters = pd.DataFrame(raters)
        feat_compare = pd.concat([feat_compare,raters], axis=1)
        feat_compare['index1'] = feat_compare.index
        feat_compare=feat_compare.reset_index()
        feat_compare= feat_compare.astype(float) # This is the final Data Frame from which we assess ICC for this radiomics feature
        
        # Evaluate ICC (Pingouin uses the same code than that in R to evaluate ICCs)
        icc = pg.intraclass_corr(data=feat_compare,targets='index1', raters='rater',ratings=new_hd[ii]).round(3)
        
        # The ICC value for this radiomics feature is stored
        iccList.append(icc.iloc[5,2])
        
        # Display ICC extraction info
        print('ICC for '+label+' '+file+' and '+new_hd[ii]+' = '+str(icc.iloc[5,2]))
        
        ii+=1
    
    # Save as .mat file the iccList including all radiomics features for a given set of preprocessing parameters
    sio.savemat(join(mypathsave,label+'.mat'),{'icc':iccList})

# Save radiomics features names list
npobjarray = np.asarray(new_hd.iloc[:],dtype=np.object)
sio.savemat(join(mypathsave+'featureNames.mat'),{'names':npobjarray})