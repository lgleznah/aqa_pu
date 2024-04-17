#!/usr/bin/env python
# coding: utf-8

# Best kNN value experiments

import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import traceback
import os
import gc

from pu.feature_extractors.extractors import ViTExtractor, AutoencoderExtractor
from pu.data.loaders import CSVLoader, SingleCSVLoader, SingleCSVWithTestLoader, FullCSVLoader
from pu.data.pu_builder import build_pu_data, pn_test_split
from pu.metrics import aul_pu

from pu.algorithms.pu_algorithms import IterativeClassifierAlgorithm, ProbTagging, NonNegativePU
from pu.algorithms.negative_detectors import NaiveDetector, KNNDetector, KMeansDetector
from pu.algorithms.stop_criterion import StopOnMetricDrop, NonStop

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def ava_splits(features_dict, extractor, quantile):
    features = features_dict[f"{extractor}__ava"]
    
    X_train, X_val, X_test, y_train, y_val, y_test, y_test_pu = build_pu_data(
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

    return X_train, X_val, X_test, y_train, y_val, y_test, y_test_pu

def aadb_splits(features_dict, extractor, quantile):
    features_train = features_dict[f"{extractor}__aadb_train"]
    features_val = features_dict[f"{extractor}__aadb_val"]
    features_test = features_dict[f"{extractor}__aadb_test"]
    
    X_train, _, _, y_train, _, _, _ = build_pu_data(
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

    _, X_val, _, _, y_val, _, _ = build_pu_data(
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

    _, _, X_test, y_test, y_test_pu = pn_test_split(
        features_test, 
        lambda row, df: row['label'] > quantile, 
        lambda row, df: row['label'] >= 0.5, 
        1.0, 
        random_state=1234
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, y_test_pu

# LAION-AES 6.5 is considered to only contain highly-aesthetic images. No need for unlabeled examples.
def laion_splits(features_dict, extractor, quantile):
    features = features_dict[f"{extractor}__laion_aes"]

    X_train, X_val, X_test, y_train, y_val, y_test, y_test_pu = build_pu_data(
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

    return X_train, X_val, X_test, y_train, y_val, y_test, y_test_pu

def full_ava_test(features_dict, extractor, quantile):
    features = features_dict[f"{extractor}__ava"]
    
    _, _, X_test, y_test, y_test_pu = pn_test_split(
        features, 
        lambda row, df: row['VotesMean'] > quantile, 
        lambda row, df: row['VotesMean'] >= 5.0, 
        1.0, 
        random_state=1234
    )
    
    return _, _, X_test, _, _, y_test, y_test_pu
    
def full_aadb_test(features_dict, extractor, quantile):
    features_train = features_dict[f"{extractor}__aadb_train"]
    features_val = features_dict[f"{extractor}__aadb_val"]
    features_test = features_dict[f"{extractor}__aadb_test"]
    
    features = pd.concat([features_train, features_val, features_test])
    
    _, _, X_test, y_test, y_test_pu = pn_test_split(
        features, 
        lambda row, df: row['label'] > quantile, 
        lambda row, df: row['label'] >= 0.5, 
        1.0, 
        random_state=1234
    )
    
    return _, _, X_test, _, _, y_test, y_test_pu

def get_laion_train_func(train_ds_func):
    def train_func(features_dict, extractor, quantile):
        laion_train, laion_val, laion_test, _, _, _, _ = laion_splits(features_dict, extractor, quantile)
        laion_full = np.concatenate([laion_train, laion_val, laion_test], axis=0)
        
        other_train, other_val, _, _, _, _, _ = train_ds_func(features_dict, extractor, quantile)
        other_full = np.concatenate([other_train, other_val], axis=0)
        
        ds_full = np.concatenate([laion_full, other_full], axis=0)
        labels = np.concatenate([np.ones(len(laion_full)), np.zeros(len(other_full))])
        X_train, X_val, y_train, y_val = train_test_split(ds_full, labels, test_size=0.2, random_state=1234, shuffle=True, stratify=labels)
	
        return X_train, X_val, None, y_train, y_val, None, None
	
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
    extractor_col, reliablepos_col, classifier_col, detector_col, bal_acc_col, acc_col, f1_col, aul_col = [], [], [], [], [], [], [], []
    y_true_col, y_pred_col = [], []

    exp_file = f"{exp_name}_results.csv"
    #if os.path.exists(exp_file):
        #return

    for extractor in extractors:
        for relpos_thresh in relpos:
            for cls in classifiers_and_args:
                for detector in neg_detectors:
                    X_train, X_val, _, y_train, y_val, _, _ = train_ds_func(features, extractor, relpos_thresh)
                    negative_detector = neg_detectors[detector][0](**neg_detectors[detector][1])
                    classifier = create_iterative(negative_detector, *classifiers_and_args[cls])
                    classifier.fit(X_train, y_train, X_val, y_val)
                    y_pred = classifier.predict(X_val)
                    y_true_col.append(y_val.tolist())
                    y_pred_col.append(classifier.predict_proba(X_val).tolist())
                    
                    bal_acc = balanced_accuracy_score(y_val, y_pred)
                    acc = accuracy_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred)
                    aul = aul_pu(y_val, y_pred)

                    extractor_col.append(extractor)
                    reliablepos_col.append(relpos_thresh)
                    classifier_col.append(cls)
                    detector_col.append(detector)
                    bal_acc_col.append(bal_acc)
                    acc_col.append(acc)
                    f1_col.append(f1)
                    aul_col.append(aul)
                    gc.collect()

    df = pd.DataFrame.from_dict({
        'extractor': extractor_col,
        'reliable_positive_threshold': reliablepos_col,
        'classifier': classifier_col,
        'detector': detector_col,
        'balanced_accuracy': bal_acc_col,
        'accuracy': acc_col,
        'f1': f1_col,
        'aul': aul_col,
        'y_true': y_true_col,
        'y_pred': y_pred_col
    })

    df.to_csv(exp_file)

def run_all_experiments(features):

    # AVA and AADB quantiles
    ava_quantiles =  [5.386517, 5.475771, 5.566116, 5.660284, 5.758871, 5.865385, 5.987416, 6.129032, 6.307692, 6.574194, 7.069421]
    aadb_quantiles = [0.5, 0.55, 0.55, 0.6, 0.6, 0.65, 0.65, 0.7, 0.75, 0.8, 0.9]

    classifiers = {
        'logistic': (LogisticRegression, {'max_iter':10000, 'n_jobs':-1, 'random_state':1234}),
    }
    
    negative_detectors = {
        f'knn-{k}': (KNNDetector, {'frac': 0.1, 'k': k}) for k in range(5, 201, 5)
    }

    negative_detectors_small_k = {
        f'knn-{k}': (KNNDetector, {'frac': 0.1, 'k': k}) for k in range(1, 5)
    }
    
    extractors = ['clip-ViT-L-14']

    # kNN experiments
    run_experiment(features, get_laion_train_func(ava_splits), ava_splits, 'laion+ava_ava_knn_selection', extractors, [0.5], classifiers, negative_detectors)
    run_experiment(features, get_laion_train_func(ava_splits), ava_splits, 'laion+ava_ava_knn_selection_small', extractors, [0.5], classifiers, negative_detectors_small_k)


def drop_not_features(df):
    drop_columns = [col_name for col_name in df.columns if "__feature__" not in col_name]
    return df.drop(columns=drop_columns)

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

            '''
            if (extractor == 'clip-ViT-L-14'):
                negatives = np.load('image_examples/positives/positives.npy')
                features_np = drop_not_features(features).to_numpy()
                for i in range(negatives.shape[0]):
                    for j in range(features.shape[0]):
                        if np.array_equal(negatives[i,:], features_np[j,:]):
                            print(data.iloc[j]['id'], data.iloc[j]['VotesMean'])
                            shutil.copy2(data.iloc[j]['id'], 'image_examples/positives')
            '''

            df = pd.concat([data.drop(columns=[path_col]), features.drop(columns=["id"])], axis=1)
            all_features[featureset_name] = df

    run_all_experiments(all_features)

if __name__ == "__main__":
    main()
