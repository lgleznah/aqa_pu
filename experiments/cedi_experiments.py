#!/usr/bin/env python
# coding: utf-8

# Experiments for the CEDI conference

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import traceback
import os
import gc

from pu.feature_extractors.extractors import ViTExtractor, AutoencoderExtractor
from pu.data.loaders import CSVLoader, SingleCSVLoader, SingleCSVWithTestLoader, FullCSVLoader
from pu.data.pu_builder import build_pu_data, pn_test_split

from pu.algorithms.pu_algorithms import IterativeClassifierAlgorithm, ProbTagging, NonNegativePU
from pu.algorithms.negative_detectors import NaiveDetector, KNNDetector, KMeansDetector
from pu.algorithms.stop_criterion import StopOnMetricDrop, NonStop

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# Having verified the proper behaviour of the implemented algorithms, it is now a good time to verify how these work when training and evaluating with and against AVA, LAION-AES and AADB.
# 
# Specifically, we will try two-step algorithms with the following base classifiers and negative detectors:
# 
# - Logistic Regression, RBF SVM, Gaussian Process, Gradient Boosting
# - K-means, KNN 
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

# ## Experiment running
# ### Dataset partitioning

# With the features extracted, now we can apply the three different algorithms. First, each dataset has a different way of partitioning, so this must be taken into account.


def ava_splits(features_dict, extractor, quantile):
    features = features_dict[f"{extractor}__ava"]
    
    X_train, X_val, X_test, y_train, y_val, y_test = build_pu_data(
        features,
        frac=1.0,
        move_to_unlabeled_frac=0.0,
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
        move_to_unlabeled_frac=0.0,
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
        move_to_unlabeled_frac=0.0,
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

def full_ava_test(features_dict, extractor, quantile):
    features = features_dict[f"{extractor}__ava"]
    
    _, _, X_test, y_test = pn_test_split(
        features, 
        lambda row, df: row['VotesMean'] > quantile, 
        lambda row, df: row['VotesMean'] >= 5.0, 
        1.0, 
        random_state=1234
    )
    
    return _, _, X_test, _, _, y_test
    
def full_aadb_test(features_dict, extractor, quantile):
    features_train = features_dict[f"{extractor}__aadb_train"]
    features_val = features_dict[f"{extractor}__aadb_val"]
    features_test = features_dict[f"{extractor}__aadb_test"]
    
    features = pd.concat([features_train, features_val, features_test])
    
    _, _, X_test, y_test = pn_test_split(
        features, 
        lambda row, df: row['label'] > quantile, 
        lambda row, df: row['label'] >= 0.5, 
        1.0, 
        random_state=1234
    )
    
    return _, _, X_test, _, _, y_test

def get_laion_train_func(train_ds_func):
    def train_func(features_dict, extractor, quantile):
        laion_train, laion_val, laion_test, _, _, _ = laion_splits(features_dict, extractor, quantile)
        laion_full = np.concatenate([laion_train, laion_val, laion_test], axis=0)
        
        other_train, other_val, other_test, _, _, _ = train_ds_func(features_dict, extractor, quantile)
        other_full = np.concatenate([other_train, other_val, other_test], axis=0)
        
        ds_full = np.concatenate([laion_full, other_full], axis=0)
        labels = np.concatenate([np.ones(len(laion_full)), np.zeros(len(other_full))])
        X_train, X_val, y_train, y_val = train_test_split(ds_full, labels, test_size=0.2, random_state=1234, shuffle=True, stratify=labels)
	
        return X_train, X_val, None, y_train, y_val, None
	
    return train_func


# ### Experiments

# Having defined how each dataset should be partitioned, the experiments themselves can now be run


def create_iterative(detector, classifer, classifier_kwargs):
    return IterativeClassifierAlgorithm(
        negative_detector=detector,
        stop_criterion=NonStop('aul'),
        classifier_class=classifer,
        max_iterations=20,
        verbose=True,
        classifier_kwargs=classifier_kwargs
    )


def run_experiment(features, train_ds_func, test_ds_func, exp_name, extractors, relpos, classifiers_and_args, neg_detectors):
    extractor_col, reliablepos_col, classifier_col, detector_col, bal_acc_col, acc_col, f1_col = [], [], [], [], [], [], []

    exp_file = f"{exp_name}_results.csv"
    #if os.path.exists(exp_file):
        #return

    for extractor in extractors:
        for relpos_thresh in relpos:
            for cls in classifiers_and_args:
                for detector in neg_detectors:
                    X_train, X_val, _, y_train, y_val, _ = train_ds_func(features, extractor, relpos_thresh)
                    _, _, X_test, _, _, y_test = test_ds_func(features, extractor, relpos_thresh)

                    negative_detector = neg_detectors[detector][0](**neg_detectors[detector][1])
                    classifier = create_iterative(negative_detector, *classifiers_and_args[cls])
                    classifier.fit(X_train, y_train, X_val, y_val)
                    
                    bal_acc = balanced_accuracy_score(y_test, classifier.predict(X_test))
                    acc = accuracy_score(y_test, classifier.predict(X_test))
                    f1 = f1_score(y_test, classifier.predict(X_test))

                    extractor_col.append(extractor)
                    reliablepos_col.append(relpos_thresh)
                    classifier_col.append(cls)
                    detector_col.append(detector)
                    bal_acc_col.append(bal_acc)
                    acc_col.append(acc)
                    f1_col.append(f1)
                    gc.collect()

    df = pd.DataFrame.from_dict({
        'extractor': extractor_col,
        'reliable_positive_threshold': reliablepos_col,
        'classifier': classifier_col,
        'detector': detector_col,
        'balanced_accuracy': bal_acc_col,
        'accuracy': acc_col,
        'f1': f1_col
    })

    df.to_csv(exp_file)

def run_baseline_experiments(features_dict, classifiers):
    # Get AVA and AADB features
    ava_feats = features_dict["clip-ViT-L-14__ava"]
    aadb_feats_train = features_dict["clip-ViT-L-14__aadb_train"]
    aadb_feats_val = features_dict["clip-ViT-L-14__aadb_val"]
    aadb_feats_test = features_dict["clip-ViT-L-14__aadb_test"]

    X_train_ava, X_val_ava, X_test_ava, y_train_ava, y_val_ava, y_test_ava = build_pu_data(
        ava_feats,
        frac=1.0,
        move_to_unlabeled_frac=0.0,
        val_split=0.2,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['VotesMean'] > 5.0,
        positive_fn=lambda row, df: row['VotesMean'] >= 5.0,
        test_frac=0.2,
        random_state=1234
    )

    X_train_aadb, _, _, y_train_aadb, _, _ = build_pu_data(
        aadb_feats_train,
        frac=1.0,
        move_to_unlabeled_frac=0.0,
        val_split=0,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['label'] > 0.5,
        positive_fn=lambda row, df: row['label'] >= 0.5,
        test_frac=0,
        random_state=1234
    )

    _, X_val_aadb, _, _, y_val_aadb, _ = build_pu_data(
        aadb_feats_val,
        frac=1.0,
        move_to_unlabeled_frac=0.0,
        val_split=1.0,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['label'] > 0.5,
        positive_fn=lambda row, df: row['label'] >= 0.5,
        test_frac=0,
        random_state=1234
    )

    _, _, X_test_aadb, y_test_aadb = pn_test_split(
        aadb_feats_test, 
        lambda row, df: row['label'] > 0.5, 
        lambda row, df: row['label'] >= 0.5, 
        1.0, 
        random_state=1234
    )

    X_train_laion, X_val_laion, X_test_laion, _, _, _ = laion_splits(features_dict, 'clip-ViT-L-14', 0.5)
    X_train_laion_full = np.concatenate([X_train_laion, X_val_laion, X_test_laion], axis=0)
    y_train_laion_full = np.ones(len(X_train_laion_full))

    settings_splits = {
        'AVA-AVA': (X_train_ava, X_test_ava, y_train_ava, y_test_ava),
        'AVA-AADB': (X_train_ava, X_test_aadb, y_train_ava, y_test_aadb),
        'AADB-AVA': (X_train_aadb, X_test_ava, y_train_aadb, y_test_ava),
        'AADB-AADB': (X_train_aadb, X_test_aadb, y_train_aadb, y_test_aadb),
        'LAION+AVA-AADB': (
            np.concatenate([X_train_laion_full, X_train_ava]),
            X_test_aadb,
            np.concatenate([y_train_laion_full, y_train_ava]),
            y_test_aadb
        ),
        'LAION+AADB-AVA': (
            np.concatenate([X_train_laion_full, X_train_aadb]),
            X_test_ava,
            np.concatenate([y_train_laion_full, y_train_aadb]),
            y_test_ava
        ),
    }

    # Run experiments with baseline PN classifiers
    setting_col, classifier_col, balacc_col, acc_col, f1_col = [], [], [], [], []

    for setting in settings_splits:
        for cls in classifiers:
            X_train, X_test, y_train, y_test = settings_splits[setting]
            classifier = classifiers[cls][0](**classifiers[cls][1])
            classifier.fit(X_train, y_train)

            bal_acc = balanced_accuracy_score(y_test, classifier.predict(X_test))
            acc = accuracy_score(y_test, classifier.predict(X_test))
            f1 = f1_score(y_test, classifier.predict(X_test))

            setting_col.append(setting)
            classifier_col.append(cls)
            balacc_col.append(bal_acc)
            acc_col.append(acc)
            f1_col.append(f1)
            gc.collect()

    df = pd.DataFrame.from_dict({
        'setting': setting_col,
        'classifier': classifier_col,
        'balanced_accuracy': balacc_col,
        'accuracy': acc_col,
        'f1': f1_col
    })

    df.to_csv("baselines_nomove.csv")
    

def run_all_experiments(features):

    # AVA and AADB quantiles
    ava_quantiles =  [5.386517, 5.475771, 5.566116, 5.660284, 5.758871, 5.865385, 5.987416, 6.129032, 6.307692, 6.574194, 7.069421]
    aadb_quantiles = [0.5, 0.55, 0.55, 0.6, 0.6, 0.65, 0.65, 0.7, 0.75, 0.8, 0.9]

    classifiers = {
        'logistic': (LogisticRegression, {'max_iter':10000, 'n_jobs':-1, 'random_state':1234}),
        'kneighbours': (KNeighborsClassifier, {'n_neighbors': 20}),
        'qda': (QuadraticDiscriminantAnalysis, {}),
        'naivebayes': (GaussianNB, {})
    }
    
    negative_detectors = {
        'k-means': (KMeansDetector, {'frac': 0.1, 'n_clusters': 20, 'random_state': 1234}),
        'knn': (KNNDetector, {'frac': 0.1, 'k': 20})
    }
    
    extractors = ['clip-ViT-L-14']

    # Baselines
    run_baseline_experiments(features, classifiers)
    return

    # AVA experiments
    run_experiment(features, ava_splits, ava_splits, 'ava_ava_twostep_nomove', extractors, ava_quantiles, classifiers, negative_detectors)
    run_experiment(features, ava_splits, aadb_splits, 'ava_aadb_twostep_nomove', extractors, ava_quantiles, classifiers, negative_detectors)
    
    # AADB experiments
    run_experiment(features, aadb_splits, ava_splits, 'aadb_ava_twostep_nomove', extractors, aadb_quantiles, classifiers, negative_detectors)
    run_experiment(features, aadb_splits, aadb_splits, 'aadb_aadb_twostep_nomove', extractors, aadb_quantiles, classifiers, negative_detectors)
    
    # LAION experiments
    run_experiment(features, get_laion_train_func(ava_splits), full_aadb_test, 'laion+ava_aadb_twostep_nomove', extractors, [0.5], classifiers, negative_detectors)
    run_experiment(features, get_laion_train_func(aadb_splits), full_ava_test, 'laion+aadb_ava_twostep_nomove', extractors, [0.5], classifiers, negative_detectors)
    

def main():
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

    run_all_experiments(all_features)

if __name__ == "__main__":
    main()
