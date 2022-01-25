# -*- coding: utf-8 -*-
"""
Format Radiomics Features for Analysis.
Save features in separate files according to classification categories (e.g. early vs. advanced FIGO stage)
and excluding unreproducible features prior to further analyses.

Not for clinical use.
SPDX-FileCopyrightText: 2021 Medical Physics Unit, McGill University, Montreal, CAN
SPDX-FileCopyrightText: 2021 Thierry Lefebvre
SPDX-FileCopyrightText: 2021 Peter Savadjiev
SPDX-License-Identifier: MIT
"""

import numpy as np
import os
import glob
import h5py
from sklearn.preprocessing import LabelEncoder
import scipy.io as sio
import pandas as pd

## THIS HAS TO BE REPEATED FOR TRAINING AND TESTING

## INPUT = RAD FTS FOR ONE MR SEQUENCE OR CT AND FILE WITH RAD FTS TO BE EXCLUDED FROM STEP 3 IN MATLAB
## OUTPUT = MATRIX ROWS = PATIENTS AND COLUMNS = RAD FTS AND VECTOR OF LABELS (0 VS 1)

# This path is expected to contain two files (0 and 1 corresponding to patient outcomes)
train_path = 'MYPROJECTFILEPATH/FINAL_EXTRACTIONS/'

# Get the labels (0 or 1)
train_labels = os.listdir(train_path)
train_labels.sort()

# Empty lists to hold feature matrices and labels
global_features = []
labels = []
labels_name = []

i, j, k = 0, 0, 0

# Loop over all patients from each category
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    print(training_name)

    # Get the current label
    current_label = training_name
    print("Current Label = ",current_label) # 0 or 1
    
    # Path where radiomics features csv files are
    csv_folders = os.listdir(train_path+training_name)
    
    # Loop over all csv files containing radiomics features for each patient
    for csv_folder in csv_folders: # corresponds to the MRI sequence name to be added before each feature name
        k = 0
        for file in glob.iglob(train_path+training_name+'/'+csv_folder+'/*.csv'):
            print(file)
            
            #Get radiomics features for each patient and store it in a matrix   
            global_feature = pd.read_csv(file, delimiter=',')
            global_values = global_feature.iloc[37:-1,1] # range in which radiomics features are located (might need to be updated if pyradiomics updates this)
            global_values = global_values.astype(float)
            global_values = np.array(global_values.values)
            
            # Get radiomics features names
            global_names = global_feature.iloc[37:-1,0]
            global_names = np.array(global_names.values)
            
            # Build the matrix (columns are features, rows are patients)
            global_feature = np.vstack((global_names, global_values))
            global_feature = pd.DataFrame(global_feature)
            new_hd = global_feature.iloc[0]
            new_hd.iloc[:] = csv_folder + new_hd.iloc[:].astype(str) #if multiple csv folders (add csv folder name in feature name)
            new_hd.iloc[:] = new_hd.iloc[:].astype(str)
            global_feature = global_feature[1:]
            global_feature.columns = new_hd
            global_feature.reset_index(drop = True)

            rescaled_feature = global_feature
            
            if i == 0:
                labels.append(current_label)

            if k == 0:
                global_features = rescaled_feature                
            else:
                global_features = global_features.append(rescaled_feature,ignore_index=True)
                
            print('[STATUS] processed CSV file: ',csv_folder)
            k += 1
        i += 1
        # If there are many sets of radiomics features (e.g multiple MRI sequences)
        if i == 1 and j == 0:
            export_frame = global_features
        elif j == 0:
            export_frame = pd.concat([export_frame, global_features], axis=1)
        elif i ==1 and j == 1:
            export_frame1 = global_features
        elif j == 1:
            export_frame1 = pd.concat([export_frame1, global_features], axis=1)
        elif i ==1 and j == 2:
            export_frame2 = global_features
        elif j == 2:
            export_frame2 = pd.concat([export_frame2, global_features], axis=1)
        elif i ==1 and j == 3:
            export_frame3 = global_features
        elif j == 3:
            export_frame3 = pd.concat([export_frame3, global_features], axis=1)
        elif i ==1 and j == 4:
            export_frame4 = global_features
        elif j == 4:
            export_frame4 = pd.concat([export_frame4, global_features], axis=1)
            
    print("[STATUS] processed folder: ",current_label)
    i = 0
    j += 1

# If there are many sets of radiomics features (e.g multiple MRI sequences) - UNCOMMENT/COMMENT lines as relevant
export_frame = export_frame.append(export_frame1,ignore_index=True)
export_frame = export_frame.append(export_frame2,ignore_index=True)
export_frame = export_frame.append(export_frame3,ignore_index=True)
export_frame = export_frame.append(export_frame4,ignore_index=True)

# csv_folders should be the name of each MRI sequence in a list
for modality in csv_folders: # Load unreproducible radiomics features names to be excluded (and repeat these lines for each MRI sequence)
    headmat=sio.loadmat('MYPROJECTFILEPATH/namersonRealout'+modality+'.mat')
    dropnames = np.array(headmat['namersonRealout'])[0]
    # Drop unreproducible features from the matrix
    for dropname in dropnames:
        export_frame = export_frame.drop(columns=str(dropname[0]))
        print('Dropped:   '+dropname[0])

# Save radiomics features for all patients
export_frame.to_csv('MYPROJECTFILEPATH/OUTPUT/coeff.csv',index=False) # merged features from all MRI sequence in one table

# Encode the target labels
labelsOut = labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)

# Save labels
h5f_label = h5py.File('MYPROJECTFILEPATH/OUTPUT/label.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))
h5f_label.close()

sio.savemat('MYPROJECTFILEPATH/OUTPUT/coeff.mat',{'coeff':export_frame, 'label':labelsOut})
