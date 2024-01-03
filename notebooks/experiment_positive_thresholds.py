#!/usr/bin/env python
# coding: utf-8

# # Experiments by changing the threshold of reliable positives

# In[1]:


import sys
sys.path.append("..")


# In[2]:


import numpy as np
import pandas as pd
import traceback
import os

from pu.feature_extractors.extractors import ViTExtractor, AutoencoderExtractor
from pu.data.loaders import CSVLoader, SingleCSVLoader, SingleCSVWithTestLoader, FullCSVLoader
from pu.data.pu_builder import build_pu_data

from pu.algorithms.pu_algorithms import IterativeClassifierAlgorithm, ProbTagging
from pu.algorithms.negative_detectors import NaiveDetector, KNNDetector
from pu.algorithms.stop_criterion import StopOnMetricDrop, NonStop

from sklearn import svm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# ## AVA dataset for positive and unlabeled examples

# In[3]:


def quantile_experiment(extractor, quantile):
    '''ava_loader = FullCSVLoader(
        '/srv/PU-dataset/unlabeled.csv',
        'id',
        '/srv/PU-dataset/dataset_unlabeled',
        reliable_positive_fn=lambda row, df: row['VotesMean'] > quantile,
        positive_fn=lambda row, df: row['VotesMean'] >= 5.0,
        test_frac=0.2,
        random_state=1234
    )'''
    ava_loader = FullCSVLoader(
        '/srv/PU-dataset/unlabeled.csv',
        'id',
        '/srv/PU-dataset/dataset_unlabeled'
    )
    
    ava_df = ava_loader.load_data()
    features = extractor.extract_features(ava_df['id'])
    ava_df = pd.concat([ava_df.drop(columns=['id']), features.drop(columns=['id'])], axis=1)

    X_train, X_val, X_test, y_train, y_val, y_test = build_pu_data(
        ava_df,
        frac=1.0,
        move_to_unlabeled_frac=0.5,
        val_split=0.2,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['VotesMean'] > quantile,
        positive_fn=lambda row, df: row['VotesMean'] >= 5.0,
        test_frac=0.2,
        random_state=1234
    )

    iterative_cls = IterativeClassifierAlgorithm(
        negative_detector=KNNDetector(frac=0.1, k=20),
        stop_criterion=NonStop('aul'),
        classifier_class=LogisticRegression,
        max_iterations=20,
        verbose=True,
        classifier_kwargs={'max_iter':10000, 'solver':'saga', 'n_jobs':-1, 'random_state':1234}
    )
    
    iterative_cls.fit(X_train, y_train, X_val, y_val)
    print(f'Evolution of f1 score: {iterative_cls.validation_results}')

    bal_acc = balanced_accuracy_score(y_test, iterative_cls.predict(X_test))
    acc = accuracy_score(y_test, iterative_cls.predict(X_test))
    f1 = f1_score(y_test, iterative_cls.predict(X_test))

    return bal_acc, acc, f1


# In[ ]:


extractors = ['clip-ViT-B-32', 'clip-ViT-B-16', 'clip-ViT-L-14']

# Quantiles [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
quantile_quantities = [
    5.386517,
    5.475771,
    5.566116,
    5.660284,
    5.758871,
    5.865385,
    5.987416,
    6.129032,
    6.307692,
    6.574194,
    7.069421
]

extractor_col, quantile_col, bal_acc_col, acc_col, f1_col = [], [], [], [], []

for quantile in quantile_quantities:
    for extractor in extractors:
        try:
            vit_extractor = ViTExtractor('quantile_experiments', extractor_name=extractor)
            bal_acc, acc, f1 = quantile_experiment(vit_extractor, quantile)
            extractor_col.append(extractor)
            quantile_col.append(quantile)
            bal_acc_col.append(bal_acc)
            acc_col.append(acc)
            f1_col.append(f1)

        except Exception as e:
            print(f'Fail at {quantile}, {extractor}')
            print(traceback.format_exc())

df = pd.DataFrame.from_dict({
    'extractor': extractor_col,
    'quantile': quantile_col,
    'balanced_accuracy': bal_acc_col,
    'accuracy': acc_col,
    'f1': f1_col
})

df.to_csv('quantile_threshold_vit_results.csv')


# In[ ]:


extractor_filters = [[8,16,16,32], [8,16,32,64,64], [8,16,32,64,64,128]]

# Quantiles [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
quantile_quantities = [
    5.386517,
    5.475771,
    5.566116,
    5.660284,
    5.758871,
    5.865385,
    5.987416,
    6.129032,
    6.307692,
    6.574194,
    7.069421
]

extractor_col, quantile_col, bal_acc_col, acc_col, f1_col = [], [], [], [], []

for quantile in quantile_quantities:
    for filters in extractor_filters:
        try:
            filename = f'quantile_threshold_autoencoder_results_{quantile}_{"_".join(str(i) for i in filters)}.csv'
            if (not os.path.exists(filename)):
                extractor = AutoencoderExtractor('quantile_experiments', input_shape=(256, 256, 3), filters=filters)
                bal_acc, acc, f1 = quantile_experiment(extractor, quantile)
                extractor_col.append(filters)
                quantile_col.append(quantile)
                bal_acc_col.append(bal_acc)
                acc_col.append(acc)
                f1_col.append(f1)
    
                df = pd.DataFrame.from_dict({
                    'extractor': extractor_col,
                    'quantile': quantile_col,
                    'balanced_accuracy': bal_acc_col,
                    'accuracy': acc_col,
                    'f1': f1_col
                })
                
                df.to_csv(filename)
                extractor_col, quantile_col, bal_acc_col, acc_col, f1_col = [], [], [], [], []

        except Exception as e:
            print(f'Fail at {quantile}, {extractor}')
            print(traceback.format_exc())


# In[ ]:




