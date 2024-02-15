from abc import ABC, abstractmethod

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

import numpy as np

class NegativeDetector(ABC):
    '''
    Reliable negative detector in PU data

    This is the parent class of all the reliable negative examples detector
    for positive-unlabeled problems
    '''

    @abstractmethod
    def detect_negatives(self, positives, unlabeled):
        '''
        Detect reliable negatives in unlabeled data

        This is the abstract function which must be overriden by children of NegativeDetector
        to implement all the different negative detectors.

        Children of this function should return a tuple of two values: the first one is the
        list of reliable negative examples, and the second one is the list of remaining unlabeled data
        '''
        pass

class NaiveDetector(NegativeDetector):
    '''
    Naive negative detector

    Just returns a random fraction of the unlabeled data as negative data

    Parameters
    ----------
    random_state: random seed to use for the algorithms that require it.

    frac: fraction of unlabeled data to return as "reliable" negatives
    '''
    def __init__(self, frac, random_state=1234):
        self.rng = np.random.default_rng(random_state)
        self.frac = frac

    def detect_negatives(self, positives, unlabeled):
        '''
        Naively detect negatives, returning random elements.

        Parameters
        ----------
        positives: list of positive examples
        unlabeled: list of unlabeled examples
        
        Returns
        -------
        (negative, remaining): a tuple of two values, containing the reliable negatives
                           and the remaining unlabeled examples, respectively.
        '''
        amount = int(len(unlabeled) * self.frac)
        idxs = self.rng.permutation(len(unlabeled))[:amount]

        return unlabeled[idxs], np.delete(unlabeled, idxs, axis=0)
    
class KNNDetector(NegativeDetector):
    '''
    KNN-based reliable negative detector

    All unlabeled examples are ranked by their distance to the k-nearest positives. The examples
    furthest away are selected as reliable negatives. Idea from "Reliable Negative Extracting Based 
    on kNN for Learning from Positive and Unlabeled Examples", but it also has parallels with
    "Improving Positive Unlabeled Learning: Practical AUL Estimation and New Training Method for 
    Extremely Imbalanced Data Sets"

    Parameters
    ----------

    frac: fraction of unlabeled data to return as "reliable" negatives
    k: number of positive neighbours to measure distance against.
    '''
    def __init__(self, frac, k):
        self.frac = frac
        self.k = k

    def detect_negatives(self, positives, unlabeled):
        '''
        Detect negatives based on distance to the k-nearest positives.

        Parameters
        ----------
        positives: list of positive examples
        unlabeled: list of unlabeled examples
        
        Returns
        -------
        (negative, remaining): a tuple of two values, containing the reliable negatives
                           and the remaining unlabeled examples, respectively.
        '''
        neighbour_detector = NearestNeighbors(n_neighbors=self.k)
        neighbour_detector.fit(positives)

        unlabeled_distances = neighbour_detector.kneighbors(unlabeled)[0]
        average_distances = np.mean(unlabeled_distances, axis=1)

        # Rank distances and return reliable negatives
        sorted_distances_idxs = np.argsort(average_distances)[::-1]
        amount = int(len(unlabeled) * self.frac)
        idxs = sorted_distances_idxs[:amount]

        return unlabeled[idxs], np.delete(unlabeled, idxs, axis=0)
    
class KMeansDetector(NegativeDetector):
    '''
    K-means based reliable negative detector

    All examples are clustered using k-means, and each cluster is marked as positive or negative according to the
    proportion of positive samples therein. Reliable negatives are selected from the negative clusters further
    away from the positives. This technique was proposed in "Learning from Positive and Unlabelled Examples
    Using Maximum Margin Clustering", but the algorithm is described loosely, so this is an interpretation.

    Parameters
    ----------
    frac: fraction of unlabeled data to return as "reliable" negatives
    n_clusters: number of clusters to cluster the data into
    random_state: random seed to use for the algorithms that require it.
    '''
    def __init__(self, frac, n_clusters, random_state=1234):
        self.frac = frac
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.rng = np.random.default_rng(self.random_state)
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def detect_negatives(self, positives, unlabeled):
        '''
        Detect negatives based on distance to positive k-means clusters

        Parameters
        ----------
        positives: list of positive examples
        unlabeled: list of unlabeled examples
        
        Returns
        -------
        (negative, remaining): a tuple of two values, containing the reliable negatives
                           and the remaining unlabeled examples, respectively.
        '''
        # Fit clustering algorithm and compute whether each cluster is positive or not
        all_elements = np.concatenate([positives, unlabeled])
        self.clusterer.fit(all_elements)
        positive_centroids = []
        negative_centroids = []

        cluster_proportions = []

        for i in range(self.n_clusters):
            cluster_indices = np.argwhere(self.clusterer.labels_ == i)
            positive_proportion = np.sum(cluster_indices < len(positives)) / len(cluster_indices)
            cluster_proportions.append(positive_proportion)

        # Sort cluster positive proportions. Negative clusters will be the 50% of clusters with the
        # smallest proportion
        cluster_proportions_sorted = np.argsort(cluster_proportions)
        i = 0
        while (i < len(cluster_proportions_sorted)):
            if (i < len(cluster_proportions_sorted) / 2):
                negative_centroids.append(self.clusterer.cluster_centers_[cluster_proportions_sorted[i]])
            else:
                positive_centroids.append(self.clusterer.cluster_centers_[cluster_proportions_sorted[i]])
            
            i += 1


        # Rank each negative centroid by its distance to positive centroids
        avg_distance_to_positives = [np.mean([np.linalg.norm(neg_cent - pos_cent) 
                                              for pos_cent in positive_centroids]) 
                                              for neg_cent in negative_centroids
                                    ]
        
        cluster_ranking = np.argsort(avg_distance_to_positives)[::-1]

        # Select reliable negatives from progresively closer clusters. Care is taken not to
        # select known positives.
        total_to_select = int(len(unlabeled) * self.frac)
        print(f"Total to select: {total_to_select}")
        unlabeled_selected = []
        idxs_to_delete = []
        cluster_ranking_idx = 0

        while(cluster_ranking_idx < len(cluster_ranking)):
            cluster_idx = cluster_ranking[cluster_ranking_idx]
            print(f"Cluster index: {cluster_idx}")
            cluster_indices = np.argwhere(self.clusterer.labels_ == cluster_idx)
            print(f"Elements of this cluster: {cluster_indices}, {len(cluster_indices)} elements")
            cluster_indices_unlabeled = cluster_indices[cluster_indices < len(positives)]
            print(f"Unlabeled elements: {cluster_indices_unlabeled}, {len(cluster_indices_unlabeled)} elements")
            take_in_cluster = min(total_to_select, len(cluster_indices_unlabeled))
            print(f"Elements to take: {take_in_cluster}")
            random_idxs = self.rng.permutation(cluster_indices_unlabeled)
            unlabeled_selected.append(np.take(all_elements, random_idxs[:take_in_cluster], axis=0))
            idxs_to_delete.extend(random_idxs)
            total_to_select -= take_in_cluster
            if total_to_select == 0:
                break
            cluster_ranking_idx += 1
        
        return np.concatenate(unlabeled_selected, axis=0), np.delete(all_elements, idxs_to_delete, axis=0)