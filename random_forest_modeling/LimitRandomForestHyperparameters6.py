# -*- coding: utf-8 -*-
"""
Prior to feature selection and modeling, limit the hyperparameters of random forests
to avoid overfitting (tree depth, number of trees, and max # of fts to split at every node)

Not for clinical use.
SPDX-FileCopyrightText: 2021 Medical Physics Unit, McGill University, Montreal, CAN
SPDX-FileCopyrightText: 2021 Thierry Lefebvre
SPDX-FileCopyrightText: 2021 Peter Savadjiev
SPDX-License-Identifier: MIT
"""

# ONLY ON THE TRAINING DATASET
# THIS WILL PRODUCE FIGURES LIKE THOSE IN FIG 5.6 IN THESIS

# import required packages
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import  RandomForestClassifier
import seaborn as sns
from scipy.stats import zscore
from collections import OrderedDict

# Generate random seed
seed = np.random.randint(1,50)

# import labels
h5f_label = h5py.File('MYPROJECTFILEPATH/OUTPUT/label.h5', 'r')
global_labels_string = h5f_label['dataset_1']
global_labels = np.array(global_labels_string)
h5f_label.close()

# import radiomics features
global_features = pd.read_csv('MYPROJECTFILEPATH/OUTPUT/coeff.csv')
global_features = global_features.apply(zscore) # zscore normalize features
global_featuresInit = np.array(global_features.values)
global_features = np.array(global_features.values)

# Radiomics features 2D array = X and labels = y
X = global_features
y = global_labels

### MINIMIZE NUMBER OF TREES ###
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(oob_score=True,
                               max_features="sqrt",
                               random_state=seed)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(max_features='log2',
                               oob_score=True,
                               random_state=seed)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(max_features=None,
                               oob_score=True,
                               random_state=seed))
]

# adapted from scikit-learn.org/ (Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830, 2011)
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# range of number of trees to assess
min_estimators = 2
max_estimators = 100

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # out-of-bag error
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"]="Arial"
plt.rcParams['font.size'] = 16
fig = plt.figure()
ax = plt.subplot()

# OOB error rate vs n_estimators plot
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators",fontfamily='sans-serif',fontsize=13)
plt.ylabel("OOB error rate",fontfamily='sans-serif',fontsize=13)
plt.legend(loc="best",frameon=False)

sns.despine(fig=fig, ax=ax,right = True)
plt.show()

fig.savefig('imagesOut/OOBnestimators.png',dpi=600)


### MINIMIZE TREE DEPTH ###
ensemble_clfs1 = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(oob_score=True,
                               max_features="sqrt",
                               n_estimators = 30, #adjust based on findings from previous section
                               random_state=seed)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(max_features='log2',
                               n_estimators = 30, #adjust based on findings from previous section
                               oob_score=True,
                               random_state=seed)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(max_features=None,
                               n_estimators = 30, #adjust based on findings from previous section
                               oob_score=True,
                               random_state=seed))
]

error_rate1 = OrderedDict((label, []) for label, _ in ensemble_clfs1)

# range of tree depths to assess
min_depth = 2
max_depth = 30

for label, clf in ensemble_clfs1:
    for i in range(min_depth, max_depth + 1):
        clf.set_params(max_depth=i)
        clf.fit(X, y)
        oob_error = 1 - clf.oob_score_
        error_rate1[label].append((i, oob_error))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"]="Arial"
plt.rcParams['font.size'] = 16
fig1 = plt.figure()
ax1 = plt.subplot()

# OOB error rate vs max_depth plot
for label, clf_err in error_rate1.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_depth, max_depth)
plt.xlabel("max_depth",fontfamily='sans-serif',fontsize=13)
plt.ylabel("OOB error rate",fontfamily='sans-serif',fontsize=13)
plt.legend(loc="best",frameon=False)

sns.despine(fig=fig, ax=ax,right = True)
plt.show()

fig.savefig('imagesOut/OOBdepth.png',dpi=600)

