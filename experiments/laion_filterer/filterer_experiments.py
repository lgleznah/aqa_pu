#!/usr/bin/env python
# coding: utf-8

# Experiments for the PU article

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
from sklearn.cluster import MiniBatchKMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

import tensorflow as tf

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
    
    X_train, X_val, X_test, y_train, y_val, y_test, y_test_pu = build_pu_data(
        features,
        frac=1.0,
        move_to_unlabeled_frac=0.0,
        val_split=0.2,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['VotesMean'] > quantile,
        positive_fn=lambda row, df: row['VotesMean'] >= 5.0,
        test_frac=0.2,
        input_mode='features',
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
        input_mode='features',
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
        input_mode='features',
        random_state=1234
    )

    _, _, X_test, y_test, y_test_pu = pn_test_split(
        features_test, 
        lambda row, df: row['label'] > quantile, 
        lambda row, df: row['label'] >= 0.5, 
        1.0, 
        input_mode='features',
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
        reliable_positive_fn=lambda row, df: row['AESTHETIC_SCORE'] > quantile,
        positive_fn=lambda row, df: row['AESTHETIC_SCORE'] >= quantile,
        test_frac=0.2,
        input_mode='features',
        random_state=1234
    )

    # A quantile can be used to remove images below a certain threshold. This can
    # help limiting the size of LAION for its experiments
    X_train = X_train[y_train == 1]
    y_train = y_train[y_train == 1]

    X_val = X_val[y_val == 1]
    y_val = y_val[y_val == 1]

    X_test = X_test[y_test == 1]
    y_test = y_test[y_test == 1]

    return X_train, X_val, X_test, y_train, y_val, y_test, y_test_pu

# CIMA is considered to only contain highly-aesthetic images. No need for unlabeled examples.
def cima_splits(features_dict, extractor, quantile):
    features = features_dict[f"{extractor}__cima"]

    X_train, X_val, X_test, y_train, y_val, y_test, y_test_pu = build_pu_data(
        features,
        frac=1.0,
        move_to_unlabeled_frac=0,
        val_split=0.2,
        val_split_positive='same',
        reliable_positive_fn=lambda row, df: row['ranking'] > quantile,
        positive_fn=lambda row, df: row['ranking'] >= quantile,
        test_frac=0.2,
        input_mode='features',
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
        input_mode='features',
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
        input_mode='features',
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

def get_cima_train_func(train_ds_func):
    def train_func(features_dict, extractor, quantile):
        cima_train, cima_val, cima_test, _, _, _, _ = cima_splits(features_dict, extractor, quantile)
        cima_full = np.concatenate([cima_train, cima_val, cima_test], axis=0)
        
        other_train, other_val, _, _, _, _, _ = train_ds_func(features_dict, extractor, quantile)
        other_full = np.concatenate([other_train, other_val], axis=0)
        
        ds_full = np.concatenate([cima_full, other_full], axis=0)
        labels = np.concatenate([np.ones(len(cima_full)), np.zeros(len(other_full))])
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
        max_iterations=10,
        verbose=True,
        classifier_kwargs=classifier_kwargs
    )

def create_mlp_nnpu_from_layer_sizes(layer_sizes, positive_prior, loss_fn, compile_kwargs, fit_kwargs):
    model = tf.keras.models.Sequential()
    for layer_number, size in enumerate(layer_sizes):
        activation = "sigmoid" if layer_number == len(layer_sizes) - 1 else "relu"
        model.add(tf.keras.layers.Dense(size, activation=activation))

    return NonNegativePU(
        model,
        positive_prior,
        loss_fn,
        compile_kwargs,
        fit_kwargs
    )

def create_probtagging(knn_num_samples, classifier_class, num_classifiers, cls_args, cls_kwargs, positive_prior):
    return ProbTagging(
        knn_num_samples,
        classifier_class,
        num_classifiers,
        cls_args,
        cls_kwargs,
        positive_prior,
        verbose=True
    )

def create_classifer(cls_name, subclassifier=None, subclassifier_kwargs=None, prior=0.7):
    match cls_name:
        case 'tsa':
            return create_iterative(
                KNNDetector(frac=0.1, k=15),
                subclassifier,
                subclassifier_kwargs,
            )
        
        # Using the MNIST setup from the paper causes it to severely overfit
        case 'nnpu':
            return create_mlp_nnpu_from_layer_sizes(
                [1],
                prior,
                tf.keras.losses.BinaryCrossentropy(),
                {'optimizer': tf.keras.optimizers.Adam()},
                {'epochs': 200, 'callbacks': [tf.keras.callbacks.ReduceLROnPlateau()]}
            )
        
        case 'pt':
            return create_probtagging(
                knn_num_samples=18,
                classifier_class=HistGradientBoostingClassifier,
                num_classifiers=50,
                cls_args=[],
                cls_kwargs={'random_state':1234},
                positive_prior=0.7
            )
        
        case 'logistic':
            return LogisticRegression(max_iter=10000, n_jobs=-1, random_state=1234)
        
        case 'knn-5':
            return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        
        case 'knn-9':
            return KNeighborsClassifier(n_neighbors=9, n_jobs=-1)
        
        case 'knn-19':
            return KNeighborsClassifier(n_neighbors=19, n_jobs=-1)
        
        case 'nb':
            return GaussianNB()
        
        case 'svm':
            return CalibratedClassifierCV(estimator=LinearSVC(max_iter=10000, dual='auto', random_state=1234))

        case 'rf':
            return RandomForestClassifier(random_state=1234)
        
        case 'hgbc':
            return HistGradientBoostingClassifier(random_state=1234)

def run_experiment(features, train_ds_func, test_ds_func, exp_name, percentiles, train_relpos, test_relpos, cls_names):
    reliablepos_col, classifier_col, subclass_col, prior_col  = [], [], [], []
    bal_acc_col, acc_col, f1_col, cmat_col = [], [], [], []
    y_true_col, y_true_pu_col, y_pred_col = [], [], []

    exp_file = f"{exp_name}_results.csv"
    if os.path.exists(exp_file):
        return
    
    for real_percentile, train_thresh, test_thresh in zip(percentiles, train_relpos, test_relpos):
        for cls in cls_names:

            # Subclassifiers for TSA. Other algorithms do not have subclassifiers
            subclassifiers = {'none': (None, None)} if cls != 'tsa' else {
                'logistic': (LogisticRegression, {'max_iter':10000, 'n_jobs':-1, 'random_state':1234}),
                'knn-5': (KNeighborsClassifier, {'n_neighbors': 5, 'n_jobs':-1}),
                'knn-9': (KNeighborsClassifier, {'n_neighbors': 9, 'n_jobs':-1}),
                'knn-19': (KNeighborsClassifier, {'n_neighbors': 19, 'n_jobs':-1}),
                'nb': (GaussianNB, {}),
                'rf': (RandomForestClassifier, {'n_jobs':-1, 'random_state':1234}),
                'svm': (CalibratedClassifierCV, {'estimator': LinearSVC(max_iter=10000, dual='auto', random_state=1234)})
            }

            # Try different priors for NNPU
            priors = [0.7] if cls != 'nnpu' else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 

            for subclassifier_name in subclassifiers:
                for prior in priors:
                    X_train, X_val, _, y_train, y_val, _, _ = train_ds_func(features, 'clip-ViT-L-14', train_thresh)
                    _, _, X_test, _, _, y_test, y_test_pu = test_ds_func(features, 'clip-ViT-L-14', test_thresh)
                    subclasifier = subclassifiers[subclassifier_name]
                    classifier = create_classifer(cls, subclasifier[0], subclasifier[1], prior)

                    if cls in ['tsa', 'nnpu']:
                        classifier.fit(X_train, y_train, X_val, y_val)
                    
                    else:
                        classifier.fit(X_train, y_train)

                    y_pred_proba = classifier.predict_proba(X_test)
                    if (len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2):
                        y_pred_proba = y_pred_proba[:,1]
                    y_pred = (y_pred_proba > 0.5).astype(int)

                    reliablepos_col.append(real_percentile)
                    classifier_col.append(cls)
                    subclass_col.append(subclassifier_name)
                    prior_col.append(prior)

                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    cmat = confusion_matrix(y_test, y_pred)

                    bal_acc_col.append(bal_acc)
                    acc_col.append(acc)
                    f1_col.append(f1)
                    cmat_col.append(cmat.tolist())

                    y_true_col.append(y_test.tolist())
                    y_pred_col.append(y_pred_proba.tolist())
                    y_true_pu_col.append(y_test_pu.tolist())
                    gc.collect()

    df = pd.DataFrame.from_dict({
        'percentile_threshold': reliablepos_col,
        'classifier': classifier_col,
        'subclassifier': subclass_col,
        'positive_prior': prior_col,
        'balanced_accuracy': bal_acc_col,
        'accuracy': acc_col,
        'f1': f1_col,
        'confusion_matrix': cmat_col,
        'y_true': y_true_col,
        'y_pred': y_pred_col,
        'y_true_pu': y_true_pu_col
    })

    df.to_csv(exp_file)
    

def run_all_experiments(features):

    # LAION quantiles
    laion_quantiles = [0.0]
    laion_percentiles = [0]
    classifiers = ['tsa', 'nnpu']
    
    
    # LAION experiments
    run_experiment(features, get_laion_train_func(ava_splits), ava_splits, 'laion_filter+ava_ava', laion_percentiles, laion_quantiles, [10.0] * len(laion_quantiles), classifiers)
    run_experiment(features, get_laion_train_func(ava_splits), aadb_splits, 'laion_filter+ava_aadb', laion_percentiles, laion_quantiles, [10.0] * len(laion_quantiles), classifiers)
    run_experiment(features, get_laion_train_func(aadb_splits), ava_splits, 'laion_filter+aadb_ava', laion_percentiles, laion_quantiles, [10.0] * len(laion_quantiles), classifiers)
    run_experiment(features, get_laion_train_func(aadb_splits), aadb_splits, 'laion_filter+aadb_aadb', laion_percentiles, laion_quantiles, [10.0] * len(laion_quantiles), classifiers)


def drop_not_features(df):
    drop_columns = [col_name for col_name in df.columns if "__feature__" not in col_name]
    return df.drop(columns=drop_columns)

def filter_laion(all_features):
    features_numpy = np.concatenate([drop_not_features(df).to_numpy() for df in all_features.values()])
    ds_labels = np.concatenate([
        [0] * len(all_features['clip-ViT-L-14__ava']),
        [1] * len(all_features['clip-ViT-L-14__aadb_train']),
        [1] * len(all_features['clip-ViT-L-14__aadb_val']),
        [1] * len(all_features['clip-ViT-L-14__aadb_test']),
        [2] * len(all_features['clip-ViT-L-14__laion_aes']),
        [3] * len(all_features['clip-ViT-L-14__cima']),
    ])
    laion_offset = len(all_features['clip-ViT-L-14__ava']) + len(all_features['clip-ViT-L-14__aadb_train']) + len(all_features['clip-ViT-L-14__aadb_val']) + len(all_features['clip-ViT-L-14__aadb_test'])
    model = MiniBatchKMeans(n_clusters=4, random_state=1234, batch_size=524288).fit(features_numpy)
    ava_max_count = -1
    laion_idxs = None
    for i in range(4):
        cluster_idxs = (model.labels_ == i)
        ds_labels_cluster = ds_labels[cluster_idxs]
        ava_amount = sum(ds_labels_cluster == 0)
        if ava_amount > ava_max_count:
            ava_max_count = ava_amount
            laion_idxs = ds_labels_cluster[ds_labels_cluster == 2]

    all_features['clip-ViT-L-14__laion_aes'] = all_features['clip-ViT-L-14__laion_aes'].iloc[laion_idxs - laion_offset]
    return all_features

def main():
    extractors = ['clip-ViT-L-14']

    dataset_params = {
        'ava': ['/srv/PU-dataset/unlabeled.csv', 'id', '/srv/PU-dataset/dataset_unlabeled', None],
        'aadb_train': ['/srv/aadb/train.csv', 'path', '/srv/aadb', None],
        'aadb_val': ['/srv/aadb/validation.csv', 'path', '/srv/aadb', None],
        'aadb_test': ['/srv/aadb/testnew.csv', 'path', '/srv/aadb', None],
        'laion_aes': ['/srv/PU-dataset/positive.csv', 'path', '/srv/PU-dataset/dataset_positive', None],
        'cima': ['/srv/PU-dataset/cima.csv', 'path', '/srv/cima', 'ඞ']
    }

    all_features = {}

    for extractor in extractors:
        for dataset in dataset_params:
            featureset_name = f"{extractor}__{dataset}"
            loader = FullCSVLoader(*dataset_params[dataset][:-1])
            feature_extractor = ViTExtractor(extractor_name=extractor, experiment_name=featureset_name, extractor_source='transformers')

            path_col = dataset_params[dataset][1]
            data = loader.load_data(sep=dataset_params[dataset][3])
            features = feature_extractor.extract_features(data[path_col])

            df = pd.concat([data.drop(columns=[path_col]), features.drop(columns=["id"])], axis=1)
            all_features[featureset_name] = df

    all_features = filter_laion(all_features)
    run_all_experiments(all_features)

if __name__ == "__main__":
    main()