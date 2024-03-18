#!/usr/bin/env python
# coding: utf-8

# # Second batch of experiments

# In[ ]:


import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import traceback
import os
import gc
import tensorflow as tf

from pu.feature_extractors.extractors import ViTExtractor, AutoencoderExtractor
from pu.data.loaders import CSVLoader, SingleCSVLoader, SingleCSVWithTestLoader, FullCSVLoader
from pu.data.pu_builder import build_pu_data, pn_test_split

from pu.algorithms.pu_algorithms import IterativeClassifierAlgorithm, ProbTagging, NonNegativePU
from pu.algorithms.negative_detectors import NaiveDetector, KNNDetector
from pu.algorithms.stop_criterion import StopOnMetricDrop, NonStop

from sklearn import svm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# Having verified the proper behaviour of the implemented algorithms, it is now a good time to verify how these work when training and evaluating with and against AVA, LAION-AES and AADB.
# 
# Specifically, we will try the following algorithms:
# 
# - Iterative classifier
# - NNPU
# - ProbTagging
# 
# And we will test these algorithms with the following combinations of datasets:
# 
# - Train with AVA, evaluate with AVA.
# - Train with AVA, evaluate with AADB.
# - Train with AADB, evaluate with AVA.
# - Train with AADB, evaluate with AADB.
# - Train with LAION (P) + AVA (U), evaluate with AADB.
# - Train with LAION (P) + AADB (U), evaluate with AVA.

# ## Feature extraction
# 
# The first step is to extract the features for all these datasets by using the three ViT models.

# In[ ]:


extractors = ['clip-ViT-B-32', 'clip-ViT-B-16', 'clip-ViT-L-14']

dataset_params = {
    'ava': ['/srv/PU-dataset/unlabeled.csv', 'id', '/srv/PU-dataset/dataset_unlabeled'],
    'aadb_train': ['/srv/aadb/train.csv', 'path', '/srv/aadb'],
    'aadb_val': ['/srv/aadb/validation.csv', 'path', '/srv/aadb'],
    'aadb_test': ['/srv/aadb/testnew.csv', 'path', '/srv/aadb'],
    'laion_aes': ['/srv/PU-dataset/positive.csv', 'path', '/srv/PU-dataset/dataset_positive']
}

all_features = {}

for extractor in extractors:
    for dataset in dataset_params:
        featureset_name = f"{extractor}__{dataset}"
        loader = FullCSVLoader(*dataset_params[dataset])
        feature_extractor = ViTExtractor(extractor_name=extractor, experiment_name=featureset_name)

        path_col = dataset_params[dataset][1]
        data = loader.load_data()
        features = feature_extractor.extract_features(data[path_col])
        df = pd.concat([data.drop(columns=[path_col]), features.drop(columns=["id"])], axis=1)
        all_features[featureset_name] = df


# ## Experiment running
# ### Dataset partitioning

# With the features extracted, now we can apply the three different algorithms. First, each dataset has a different way of partitioning, so this must be taken into account.

# In[ ]:


def ava_splits(features_dict, extractor, quantile):
    features = features_dict[f"{extractor}__ava"]
    
    X_train, X_val, X_test, y_train, y_val, y_test = build_pu_data(
        features,
        frac=1.0,
        move_to_unlabeled_frac=0.5,
        val_split=0.2,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['VotesMean'] > quantile,
        positive_fn=lambda row, df: row['VotesMean'] >= 5.0,
        test_frac=0.2,
        random_state=1234
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def aadb_splits(features_dict, extractor, quantile):
    features_train = features_dict[f"{extractor}__aadb_train"]
    features_val = features_dict[f"{extractor}__aadb_val"]
    features_test = features_dict[f"{extractor}__aadb_test"]
    
    X_train, _, _, y_train, _, _ = build_pu_data(
        features_train,
        frac=1.0,
        move_to_unlabeled_frac=0.5,
        val_split=0,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['label'] > quantile,
        positive_fn=lambda row, df: row['label'] >= 0.5,
        test_frac=0,
        random_state=1234
    )

    _, X_val, _, _, y_val, _ = build_pu_data(
        features_val,
        frac=1.0,
        move_to_unlabeled_frac=0.5,
        val_split=1.0,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['label'] > quantile,
        positive_fn=lambda row, df: row['label'] >= 0.5,
        test_frac=0,
        random_state=1234
    )

    _, _, X_test, y_test = pn_test_split(
        features_test, 
        lambda row, df: row['label'] > quantile, 
        lambda row, df: row['label'] >= 0.5, 
        1.0, 
        random_state=1234
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# LAION-AES 6.5 is considered to only contain highly-aesthetic images. No need for unlabeled examples.
def laion_splits(features_dict, extractor, quantile):
    features = features_dict[f"{extractor}__laion_aes"]

    X_train, X_val, X_test, y_train, y_val, y_test = build_pu_data(
        features,
        frac=1.0,
        move_to_unlabeled_frac=0,
        val_split=0.2,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['AESTHETIC_SCORE'] > 5.0,
        positive_fn=lambda row, df: row['AESTHETIC_SCORE'] >= 5.0,
        test_frac=0.2,
        random_state=1234
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ### Experiments

# Having defined how each dataset should be partitioned, the experiments themselves can now be run

# In[ ]:


def create_iterative(prior):
    return IterativeClassifierAlgorithm(
        negative_detector=KNNDetector(frac=0.1, k=20),
        stop_criterion=NonStop('aul'),
        classifier_class=LogisticRegression,
        max_iterations=20,
        verbose=True,
        classifier_kwargs={'max_iter':10000, 'solver':'saga', 'n_jobs':-1, 'random_state':1234}
    )

def create_probtagging(prior):
    return ProbTagging(
        knn_num_samples=20,
        classifier_class=LogisticRegression,
        num_classifiers=10,
        verbose=True,
        classifier_kwargs={'max_iter':10000},
        positive_prior=prior
    )

def create_nnpu(prior):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    return NonNegativePU(
        model=model,
        positive_prior=prior,
        loss_fn=tf.keras.losses.BinaryCrossentropy(),
        compile_kwargs={'optimizer': 'adam'},
        fit_kwargs={'epochs': 10}
    )


# In[ ]:


def run_experiment(features, train_ds_func, test_ds_func, exp_name, extractors, relpos, algos, priors):
    extractor_col, reliablepos_col, algorithm_col, prior_col, bal_acc_col, acc_col, f1_col = [], [], [], [], [], [], []

    exp_file = f"{exp_name}_results.csv"
    if os.path.exists(exp_file):
        return

    for extractor in extractors:
        for relpos_thresh in relpos:
            for algo in algos:
                exp_priors = [0.5] if algo == "iterative" else priors
                for prior in exp_priors:
                    X_train, X_val, _, y_train, y_val, _ = train_ds_func(features, extractor, relpos_thresh)
                    _, _, X_test, _, _, y_test = test_ds_func(features, extractor, relpos_thresh)
                    classifier = algos[algo](prior)
                    classifier.fit(X_train, y_train, X_val, y_val)
                    
                    bal_acc = balanced_accuracy_score(y_test, classifier.predict(X_test))
                    acc = accuracy_score(y_test, classifier.predict(X_test))
                    f1 = f1_score(y_test, classifier.predict(X_test))

                    extractor_col.append(extractor)
                    reliablepos_col.append(relpos_thresh)
                    algorithm_col.append(algo)
                    prior_col.append(prior)
                    bal_acc_col.append(bal_acc)
                    acc_col.append(acc)
                    f1_col.append(f1)
                    gc.collect()

    df = pd.DataFrame.from_dict({
        'extractor': extractor_col,
        'reliable_positive_threshold': reliablepos_col,
        'algorithn': algorithm_col,
        'prior': prior_col,
        'balanced_accuracy': bal_acc_col,
        'accuracy': acc_col,
        'f1': f1_col
    })

    df.to_csv(exp_file)
    

def run_all_experiments(features):

    # AVA and AADB quantiles
    ava_quantiles =  [5.386517, 5.475771, 5.566116, 5.660284, 5.758871, 5.865385, 5.987416, 6.129032, 6.307692, 6.574194, 7.069421]
    aadb_quantiles = [0.5, 0.55, 0.55, 0.6, 0.6, 0.65, 0.65, 0.7, 0.75, 0.8, 0.9]

    classifiers = {'probtag': create_probtagging, 'nnpu': create_nnpu}
    extractors = ['clip-ViT-B-32', 'clip-ViT-B-16', 'clip-ViT-L-14']
    priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # AVA experiments
    run_experiment(features, ava_splits, ava_splits, 'ava_ava', extractors, ava_quantiles, classifiers, priors)


# In[ ]:


run_all_experiments(all_features)


# In[ ]:




