# -*- coding: utf-8 -*-
"""
Random forest modeling for radiomics feature selection and
for classification of high-risk histopathological markers of endometrial carcinoma

Not for clinical use.
SPDX-FileCopyrightText: 2021 Medical Physics Unit, McGill University, Montreal, CAN
SPDX-FileCopyrightText: 2021 Thierry Lefebvre
SPDX-FileCopyrightText: 2021 Peter Savadjiev
SPDX-License-Identifier: MIT
"""


# import required packages
import h5py
from sklearn.metrics import confusion_matrix, auc, plot_roc_curve, roc_auc_score,roc_curve,recall_score,precision_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import  RandomForestClassifier
import seaborn as sns
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from scipy import interp
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from scipy.stats import zscore, t
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

# import labels
h5f_label = h5py.File('MYPROJECTFILEPATH/OUTPUT/label.h5', 'r')
global_labels_string = h5f_label['dataset_1']
global_labels = np.array(global_labels_string)
h5f_label.close()

# import radiomics features
global_features = pd.read_csv('MYPROJECTFILEPATH/OUTPUT/coeff.csv')
# Saving feature names for later use
feature_list = list(global_features.columns)
global_features = global_features.apply(zscore) # zscore normalize features
global_featuresInit = np.array(global_features.values)
global_features = np.array(global_features.values)


# import validation/testing labels
h5f_labelval = h5py.File('MYPROJECTFILEPATH/OUTPUT/labelVal.h5', 'r')
global_labels_stringval = h5f_labelval['dataset_1']
validation_labels = np.array(global_labels_stringval)
h5f_labelval.close()

# import validation/testing radiomics features
validation_features = pd.read_csv('MYPROJECTFILEPATH/OUTPUT/coeffVal.csv')
validation_features = validation_features.apply(zscore) # zscore normalize features
validation_featuresInit = np.array(validation_features.values)
validation_features = np.array(validation_features.values)

# Verify the shape of the radiomics features matrix and labels
print("features shape: {}".format(global_features.shape))
print("labels shape: {}".format(global_labels.shape))


### REMOVE CORRELATED FEATURES ###

X = global_features

# Handling multi-collinear features
corr = spearmanr(X).correlation
corr_linkage = hierarchy.ward(corr)

# Clusters with spearman's rho > 0.95
cluster_ids = hierarchy.fcluster(corr_linkage, 0.95, criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
     cluster_id_to_feature_ids[cluster_id].append(idx)

# Keep only one features per cluster
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

# Select uncorrelated features
global_features = global_features[:,selected_features]
validation_features = validation_features[:,selected_features]
feature_list = np.array(feature_list)
feature_list = feature_list[selected_features]


### MACHINE LEARNING MODELING (RANDOM FORESTS) ###

# Generate random seed
seed = np.random.randint(1,50)

# THESE HYPERPARAMETERS ARE SELECTED WITH LimitRandomForestHyperparameters6.py
# Insert optimized random forest hyperparameters
num_trees = 30
tree_depth = 5
max_split = 'sqrt'

# Instantiate machine learning models
models = []
models.append(('RF', RandomForestClassifier(n_estimators=num_trees,max_depth=tree_depth,max_features = max_split,random_state=seed)))

# Variables to hold the results and names
results = []
names = []
scoring = "roc_auc"

### IDENTIFY MOST IMPORTANT FEATURES ###

# Store Gini impurity-based feature importance
importancesList = []

# Resample majority class
resample = RandomUnderSampler(sampling_strategy=0.75)

# Confidence for intervals
confidence = 0.95

# Bootstraps number
bootnum = 1000

# k-fold cross validation
for name, model in models:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=bootnum)
    pipeline = Pipeline(steps=[('r',resample),('m',model)])    
    cv_results = cross_val_score(pipeline, global_features, global_labels, cv=cv, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

    if 'RF' in name:       
        # Bootstrap starts here (# is the # of bootstraps)
        for kk in range(bootnum):
            cvoth = StratifiedKFold(n_splits=5, shuffle=True,random_state=seed+kk)
            
            for i, (train, test) in enumerate(cvoth.split(global_features, global_labels)):
                model.fit(global_features[train], global_labels[train])
                if kk ==0:
                    importancesList = np.array(model.feature_importances_)
                elif kk>0:
                    importancesList = np.vstack((importancesList,np.array(model.feature_importances_)))                 

        model.fit(global_features, global_labels)
        
        # Get predictions and probabilities on the training dataset
        y_pred = cross_val_predict(pipeline, global_features, global_labels, cv=5)
        y_proba = cross_val_predict(model, global_features, global_labels, method='predict_proba',cv=5)[:, 1]

        # Validation probabilities and predictions
        y_predVal = model.predict(validation_features)
        y_probaVal =model.predict_proba(validation_features)[:, 1]
        
        # Get feature importances on the whole training dataset
        importancesList = np.vstack((importancesList,np.array(model.feature_importances_)))


# Importances across CV bootstrapped samples
stdstd = np.std(importancesList, axis=0)
meanmean = np.mean(importancesList, axis=0)
indicesAll = np.argsort(meanmean)[::-1]

indices = indicesAll[0:20]
feature_list = np.array(feature_list)

# Plot the impurity-based feature importances of the random forest
plt.figure()
plt.ylabel('Importance'); plt.xlabel('Feature'); plt.title('Feature Importances');plt.bar(range(20), meanmean[indices],color="r", yerr=stdstd[indices])
plt.xticks(range(20), feature_list[indices],rotation=50,horizontalalignment='right')
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"]="Arial"
plt.rcParams['font.size'] = 13
sns.despine(right = True)
plt.show()

### TRAIN THE RANDOM FOREST WITH THE MOST IMPORTANT FEATURES

# Most important feature names
feat_list =np.array(feature_list)
feat_list = feat_list[indicesAll]
feature_list = list(feature_list)

# Extract the 5 most important features
important_indices = [feature_list.index(feat_list[0]), feature_list.index(feat_list[1]),feature_list.index(feat_list[2]), feature_list.index(feat_list[3]),
                     feature_list.index(feat_list[4])]


glob_feat_imp = global_features[:, important_indices]
validation_feat_imp = validation_features[:, important_indices]

# Print important features
print(feat_list[0:5])

models1 = []
models1.append(('RF', RandomForestClassifier(n_estimators=num_trees,max_depth=tree_depth,max_features = max_split,random_state=seed)))
results1 = []
names1=[]

# k-fold cross validation
for name, model in models1:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats = bootnum)
    pipeline1 = Pipeline(steps=[('r',resample),('m',model)])     
    cv_results = cross_val_score(pipeline1, glob_feat_imp, global_labels, cv=cv, scoring=scoring)
    results1.append(cv_results)
    names1.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

    if 'RF' in name:
        # Run classifier with cross-validation and plot ROC curves
        meanmean_aucs = []
        meanmean_tprs = []
        
        fig, ax = plt.subplots(figsize = (8, 8))
        fig1, ax1 = plt.subplots()

        # Bootstrap
        for kk in range(bootnum):
            cvoth = StratifiedKFold(n_splits=5, shuffle=True,random_state=seed+kk)
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            
            for i, (train, test) in enumerate(cvoth.split(glob_feat_imp, global_labels)):
                model.fit(glob_feat_imp[train], global_labels[train])
                viz = []
                viz = plot_roc_curve(model, glob_feat_imp[test], global_labels[test],
                                     name='ROC fold {}'.format(i),
                                     alpha=0.3, lw=1, ax=ax1)
                ax1.get_legend().remove()
                interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)
                interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)                 

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            if kk==0:
                meanmean_tprs = np.array(mean_tpr)
            else:
                meanmean_tprs = np.vstack([meanmean_tprs,mean_tpr])
                      
            meanmean_aucs.append(aucs)

                
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b',
                label=None, alpha=.8)
        
        mean_tpr = np.mean(meanmean_tprs, axis=0)
        mean_auc = auc(mean_fpr, np.mean(meanmean_tprs, axis=0))
        youdensi=(mean_tpr-mean_fpr)
        idxYi = np.argmax(youdensi)
        sensYi = mean_tpr[idxYi]
        specYi = 1-mean_fpr[idxYi]
        
        means_sens_Yi = meanmean_tprs[:,idxYi]
        std_sens = np.std(means_sens_Yi)
        n_sens=len(means_sens_Yi)
        h_sens = std_sens * t.ppf((1 + confidence) / 2, n_sens - 1)
        start_sens = sensYi - h_sens
        enfin_sens = sensYi + h_sens
        if enfin_sens > 1:
            enfin_sens = 1                
        print('sens')
        print(start_sens, sensYi, enfin_sens)
        
        print('spec')
        print(specYi-h_sens, specYi, specYi+h_sens)
        
        
        bal_acc = (mean_tpr+np.ones_like(mean_tpr)-mean_fpr)/2
        idxMax = np.argmax(bal_acc)
        std_acc = np.std(bal_acc)
        n_acc=len(bal_acc)
        h_acc = std_acc * t.ppf((1 + confidence) / 2, n_acc - 1)
        start_acc = bal_acc[idxMax] - h_acc
        enfin_acc =bal_acc[idxMax]+ h_acc
        if enfin_acc > 1:
            enfin_acc = 1                
        print('balanced accuracy')
        print(start_acc, bal_acc[idxMax], enfin_acc)
        
        prev=np.count_nonzero(global_labels)/len(global_labels)
        PPV = means_sens_Yi*prev/(means_sens_Yi*prev+((1-specYi)*(1-prev)))
        std_PPV = np.std(PPV)
        n_PPV=len(PPV)
        h_PPV= std_PPV* t.ppf((1 + confidence) / 2, n_PPV- 1)
        start_PPV= np.mean(PPV) - h_PPV
        enfin_PPV= np.mean(PPV)+ h_PPV
        if enfin_PPV> 1:
            enfin_PPV= 1

        print('PPV')
        print(start_PPV, np.mean(PPV), enfin_PPV) 

        NPV = specYi*(1-prev)/((np.ones_like(means_sens_Yi)-means_sens_Yi)*prev+(specYi*(1-prev)))
        std_NPV = np.std(NPV)
        n_NPV=len(NPV)
        h_NPV= std_NPV* t.ppf((1 + confidence) / 2, n_NPV- 1)
        start_NPV= np.mean(NPV) - h_NPV
        enfin_NPV= np.mean(NPV)+ h_NPV
        if enfin_NPV> 1:
            enfin_NPV= 1

        print('NPV')
        print(start_NPV, np.mean(NPV), enfin_NPV)             
        
        std_auc = np.std(meanmean_aucs)
        n=len(meanmean_aucs)
        h = std_auc * t.ppf((1 + confidence) / 2, n - 1)
        start = mean_auc - h
        enfin = mean_auc + h
        if enfin > 1:
            enfin=1
        
        print('auc')
        print(start,mean_auc,enfin)
        
        ax.plot(mean_fpr, mean_tpr, color='r',
                lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01],title=r'ROC Curve, AUC = %0.2f (%0.2f, %0.2f)' % (mean_auc, start, enfin))
        ax.set_aspect('equal', adjustable='box')
        sns.despine(fig=fig, ax=ax,right = True)
        ax.set_xlabel('False Positive Rate',fontfamily='sans-serif',fontsize=16) 
        ax.set_ylabel('True Positive Rate',fontfamily='sans-serif',fontsize=16)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams['font.size'] = 16
        plt.show()
        model.fit(glob_feat_imp, global_labels)

        # Get predictions and probabilities on the training dataset
        y_pred1 = cross_val_predict(pipeline1, glob_feat_imp, global_labels, cv=5)
        y_proba1 = cross_val_predict(pipeline1, glob_feat_imp, global_labels,method='predict_proba',cv=5)[:, 1]

        # Validation probabilities and predictions
        y_predVal1 = model.predict(validation_feat_imp)
        y_probaVal1 =model.predict_proba(validation_feat_imp)[:, 1]



def evaluate_model1(predictions, probs, testLabel):#, train_predictions, train_probs):
    baseline = {}
    baseline['recall'] = recall_score(testLabel, 
                                     [1 for _ in range(len(testLabel))])
    baseline['precision'] = precision_score(testLabel, 
                                      [1 for _ in range(len(testLabel))])
    baseline['roc'] = 0.5
    
    results = {}
    results['recall'] = recall_score(testLabel, predictions)
    results['precision'] = precision_score(testLabel, predictions)
    results['roc'] = roc_auc_score(testLabel, probs,average='samples')
    base_fpr, base_tpr, _ = roc_curve(testLabel, [1 for _ in range(len(testLabel))])
    model_fpr, model_tpr, _ = roc_curve(testLabel, probs)
    fig12=[]
    ax12=[]
    fig12, ax12 = plt.subplots(figsize = (8, 8))
    
    plt.plot(base_fpr, base_tpr, 'b--')#, label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r')#, label = 'model')
    plt.title('ROC Curve, AUC = '+str(results['roc'])[0:4]);
    ax12.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    ax12.set_aspect('equal', adjustable='box')
    sns.despine(fig=fig12, ax=ax12,right = True)
    ax12.set_xlabel('False Positive Rate',fontfamily='sans-serif',fontsize=16) 
    ax12.set_ylabel('True Positive Rate',fontfamily='sans-serif',fontsize=16)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['font.size'] = 16
    plt.show()

#evaluate ROC curve of validation with most important features
evaluate_model1(y_predVal1, y_probaVal1, validation_labels)


