import sys
sys.path.append("../..")

import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numba

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from pu.feature_extractors.extractors import ViTExtractor
from pu.data.loaders import FullCSVLoader

def get_all_features(extractor):
    dataset_params = {
        'ava': ['/srv/PU-dataset/unlabeled.csv', 'id', '/srv/PU-dataset/dataset_unlabeled', None],
        'laion_aes': ['/srv/PU-dataset/positive.csv', 'path', '/srv/PU-dataset/dataset_positive', None]
    }

    all_features = []
    ava_scores = []

    for dataset in dataset_params:
        featureset_name = f"{extractor}__{dataset}"
        loader = FullCSVLoader(*dataset_params[dataset][:-1])
        feature_extractor = ViTExtractor(extractor_name=extractor, experiment_name=featureset_name, extractor_source='transformers')

        path_col = dataset_params[dataset][1]
        data = loader.load_data(sep=dataset_params[dataset][3])
        features = feature_extractor.extract_features(data[path_col]).drop(columns=["id"]).to_numpy()

        all_features.append(features)
        if dataset == 'ava':
            ava_scores = data['VotesMean'].to_numpy()

    return all_features, ava_scores


def nn_scores(features, ava_scores, extractor):
    knn = NearestNeighbors(n_neighbors=10).fit(features[0])
    ava_nns = knn.kneighbors(features[1], return_distance=True)
    distances = ava_nns[0].flatten()
    scores = np.apply_along_axis(lambda row: ava_scores[row], axis=1, arr=ava_nns[1]).flatten()
    for i in range(10):
        filename = f"{extractor}/scores_distances_scatter_rank_{i}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        idxs = np.arange(i, len(distances), 10)
        _, ava_idxs_unique = np.unique(ava_nns[1].flatten()[idxs], return_index=True)
        scores_rank, distances_rank = scores[idxs], distances[idxs]
        scores_rank, distances_rank = scores[ava_idxs_unique], distances[ava_idxs_unique]
        ax = sns.jointplot(x=scores_rank, y=distances_rank, kind='hist', xlim=[0,10], ylim=[0,80], joint_kws={'cmap': 'mako_r'}, marginal_kws={'color': 'green'})
        ax.ax_joint.set_xlabel('Scores', fontsize=16)
        ax.ax_joint.set_ylabel('Distances', fontsize=16)
        ax.ax_joint.set_xticks(range(11))
        plt.tight_layout()
        ax._figure.savefig(filename, dpi=1200)
        plt.close()

@numba.jit(parallel=True)
def furthest_neighbours(source, ava_features, n_neighbours=10, chunk_size=1000):
    chunk_distances = np.empty((chunk_size, len(ava_features)))
    max_distances = np.empty((len(source), n_neighbours))
    max_distances_idxs = np.empty((len(source), n_neighbours), dtype=np.int32)
    idx = 0

    print(chunk_distances.shape)
    print(max_distances.shape)
    print(max_distances_idxs.shape)
    
    while idx < len(source):
        print(f"{idx}/{len(source)}\r")
        remaining_in_chunk = min(chunk_size, len(source) - idx)
        for i in range(remaining_in_chunk):
            for j in numba.prange(len(ava_features)):
                source_example = source[idx+i]
                ava_example = ava_features[j]
                chunk_distances[i,j] = np.linalg.norm(ava_example - source_example)

            max_distances_idxs[idx+i] = np.argpartition(chunk_distances[i], -n_neighbours)[-n_neighbours:]
            max_distances[idx+i] = chunk_distances[i][max_distances_idxs[idx+i]]
        idx += chunk_size

    return max_distances, max_distances_idxs

def nn_scores_furthest(features, ava_scores, extractor):
    ava_features = features[0]
    laion_features = features[1]
    ava_scores = ava_scores

    filename = f"{extractor}/scores_distances_furthest.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    max_distances, max_distances_idxs = furthest_neighbours(laion_features, ava_features, n_neighbours=1000)
    _, unique_indices = np.unique(max_distances_idxs.flatten(), return_index=True)
    max_distances = max_distances.flatten()[unique_indices]
    max_distances_idxs = max_distances_idxs.flatten()[unique_indices]
    scores = ava_scores[max_distances_idxs]
    distances = max_distances

    ax = sns.jointplot(x=scores, y=distances, kind='hist', xlim=[0,10], ylim=[0,np.max(distances)+1], joint_kws={'cmap': 'mako_r'}, marginal_kws={'color': 'green'})
    ax.ax_joint.set_xlabel('Scores', fontsize=16)
    ax.ax_joint.set_ylabel('Distances', fontsize=16)
    ax.ax_joint.set_xticks(range(11))
    plt.tight_layout()
    ax._figure.savefig(filename, dpi=1200)
    plt.close()

def nn_scores_knndetector(features, ava_scores, extractor):
    neighbour_detector = NearestNeighbors(n_neighbors=10)
    neighbour_detector.fit(features[1])
    average_distances = np.mean(neighbour_detector.kneighbors(features[0])[0], axis=1)

    filename = f"{extractor}/scores_distances_knndetector.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    sorted_distances_idxs = np.argsort(average_distances)[::-1]
    ava_scores_furthest = ava_scores[sorted_distances_idxs]
    average_distances_furthest = average_distances[sorted_distances_idxs]

    print(f"Score stats for LAION reliable negative detection:")
    print(pd.Series(ava_scores_furthest).describe())

    ax = sns.jointplot(x=ava_scores_furthest, y=average_distances_furthest, kind='hist', xlim=[0,10], ylim=[0,80], joint_kws={'cmap': 'mako_r'}, marginal_kws={'color': 'green'})
    ax.ax_joint.set_xlabel('Scores', fontsize=16)
    ax.ax_joint.set_ylabel('Distances', fontsize=16)
    ax.ax_joint.set_xticks(range(11))
    plt.tight_layout()
    ax._figure.savefig(filename, dpi=1200)
    plt.close()

def nn_scores_ava_threshold(features, ava_scores, extractor):
    positives = features[0][ava_scores >= 6.13]
    unlabeled = features[0][ava_scores < 6.13]

    ava_scores_unlabeled = ava_scores[ava_scores < 6.13]

    filename = f"{extractor}/scores_distances_knndetector_ava_threshold.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    neighbour_detector = NearestNeighbors(n_neighbors=10)
    neighbour_detector.fit(positives)
    average_distances = np.mean(neighbour_detector.kneighbors(unlabeled)[0], axis=1)

    sorted_distances_idxs = np.argsort(average_distances)[::-1]
    ava_scores_furthest = ava_scores_unlabeled[sorted_distances_idxs]
    average_distances_furthest = average_distances[sorted_distances_idxs]

    print(f"Score stats for AVA reliable negative detection:")
    print(pd.Series(ava_scores_furthest).describe())

    ax = sns.jointplot(x=ava_scores_furthest, y=average_distances_furthest, kind='hist', xlim=[0,10], ylim=[0,80], joint_kws={'cmap': 'mako_r'}, marginal_kws={'color': 'green'})
    ax.ax_joint.set_xlabel('Scores', fontsize=16)
    ax.ax_joint.set_ylabel('Distances', fontsize=16)
    ax.ax_joint.set_xticks(range(11))
    plt.tight_layout()
    ax._figure.savefig(filename, dpi=1200)
    plt.close()

def nn_scores_threestep(features, ava_scores, extractor):
    knn = NearestNeighbors(n_neighbors=10).fit(features[0])
    ava_nns = knn.kneighbors(features[1], return_distance=True)
    distances = ava_nns[0].flatten()
    indices = ava_nns[1].flatten()

    # Get examples from AVA at a distance less than a threshold, removing duplicates
    threshold = 2.5
    while threshold < 8:
        distances_thresholded = (distances <= threshold)
        indices_thresholded = indices[distances_thresholded]
        _, unique_indices = np.unique(indices_thresholded, return_index=True)
        positives = features[0][unique_indices]
        unlabeled = np.delete(features[0], unique_indices, axis=0)
        ava_scores_unlabeled = np.delete(ava_scores, unique_indices, axis=0)

        filename = f"{extractor}/scores_distances_threestep_{str(threshold)}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        neighbour_detector = NearestNeighbors(n_neighbors=10)
        neighbour_detector.fit(positives)
        average_distances = np.mean(neighbour_detector.kneighbors(unlabeled)[0], axis=1)

        sorted_distances_idxs = np.argsort(average_distances)[::-1]
        ava_scores_furthest = ava_scores_unlabeled[sorted_distances_idxs]
        average_distances_furthest = average_distances[sorted_distances_idxs]

        print(f"Score stats for threestep threshold {threshold}:")
        print(pd.Series(ava_scores_furthest).describe())

        ax = sns.jointplot(x=ava_scores_furthest, y=average_distances_furthest, kind='hist', xlim=[0,10], ylim=[0,80], joint_kws={'cmap': 'mako_r'}, marginal_kws={'color': 'green'})
        ax.ax_joint.set_xlabel('Scores', fontsize=16)
        ax.ax_joint.set_ylabel('Distances', fontsize=16)
        ax.ax_joint.set_xticks(range(11))
        plt.tight_layout()
        ax._figure.savefig(filename, dpi=1200)
        plt.close()

        threshold += 0.5

def main():
    for extractor in ['facebook/dinov2-large']:
        features, ava_scores = get_all_features(extractor)
        nn_scores(features, ava_scores, extractor)
        nn_scores_knndetector(features, ava_scores, extractor)
        nn_scores_ava_threshold(features, ava_scores, extractor)
        nn_scores_threestep(features, ava_scores, extractor)
        #nn_scores_furthest(features, ava_scores, extractor)

if __name__ == "__main__":
    main()