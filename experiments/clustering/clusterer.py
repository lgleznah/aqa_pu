import sys
sys.path.append("../..")

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from pu.feature_extractors.extractors import ViTExtractor
from pu.data.loaders import FullCSVLoader

def get_all_features():
    extractor = 'clip-ViT-L-14'

    dataset_params = {
        'ava': ['/srv/PU-dataset/unlabeled.csv', 'id', '/srv/PU-dataset/dataset_unlabeled', None],
        'aadb_train': ['/srv/aadb/train.csv', 'path', '/srv/aadb', None],
        'aadb_val': ['/srv/aadb/validation.csv', 'path', '/srv/aadb', None],
        'aadb_test': ['/srv/aadb/testnew.csv', 'path', '/srv/aadb', None],
        #'laion_aes': ['/srv/PU-dataset/positive.csv', 'path', '/srv/PU-dataset/dataset_positive', None],
        'cima': ['/srv/PU-dataset/cima.csv', 'path', '/srv/cima', 'ඞ']
    }

    dataset_idx_mapping = {
        'ava': 0,
        'aadb_train': 1,
        'aadb_val': 1,
        'aadb_test': 1,
        #'laion_aes': 2,
        'cima': 2
    }

    all_features = []
    all_ds_labels = []

    for dataset in dataset_params:
        featureset_name = f"{extractor}__{dataset}"
        loader = FullCSVLoader(*dataset_params[dataset][:-1])
        feature_extractor = ViTExtractor(extractor_name=extractor, experiment_name=featureset_name, extractor_source='transformers')

        path_col = dataset_params[dataset][1]
        data = loader.load_data(sep=dataset_params[dataset][3])
        features = feature_extractor.extract_features(data[path_col]).drop(columns=["id"]).to_numpy()

        all_features.append(features)
        all_ds_labels.append([dataset_idx_mapping[dataset]]*len(features))

    return np.concatenate(all_features), np.concatenate(all_ds_labels)


def cluster_and_print_proportions(features, ds_idxs, n_clusters=15):
    idx_dataset_mapping = ['AVA', 'AADB', 'CIMA']
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=1234, batch_size=524288).fit(features)
    for i in range(n_clusters):
        cluster_elements = features[model.labels_ == i]
        element_datasets = ds_idxs[model.labels_ == i]
        cluster_counts = []
        print(f"Cluster #{i} ({len(cluster_elements)} samples) dataset proportions:")
        for ds_idx in range(len(idx_dataset_mapping)):
            cluster_elements_ds_count = len(cluster_elements[element_datasets == ds_idx])
            cluster_percentage = cluster_elements_ds_count / len(cluster_elements) * 100
            dataset_percentage = cluster_elements_ds_count / len(ds_idxs[ds_idxs == ds_idx]) * 100
            print(f"    {idx_dataset_mapping[ds_idx]:<11}: {cluster_elements_ds_count:>6} samples ({cluster_percentage:05.2f}% of cluster, {dataset_percentage:05.2f}% of dataset)")
            cluster_counts.append(cluster_elements_ds_count)

        fig = plt.pie(cluster_counts, labels=idx_dataset_mapping)
        plt.savefig(f"cluster_{i}.svg")
        plt.close()
        print("\n")

def reset_graphs():
    for file in glob.glob("*.svg"):
        os.remove(file)

def main():
    reset_graphs()
    features, ds_idxs = get_all_features()
    cluster_and_print_proportions(features, ds_idxs, n_clusters=4)

if __name__ == "__main__":
    main()