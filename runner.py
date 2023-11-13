from pu.feature_extractors.extractors import ViTExtractor, AutoencoderExtractor
from pu.data.loaders import CSVLoader
from pu.data.pu_builder import build_pu_data

from pu.algorithms.pu_algorithms import IterativeClassifierAlgorithm
from pu.algorithms.negative_detectors import NaiveDetector
from pu.algorithms.stop_criterion import StopOnMetricDrop

import os
from sklearn import svm

# ¿Quizá usar sólo Laion?
def main():
    positive_loader = CSVLoader('/srv/PU-dataset/positive.csv', 'path', '/srv/PU-dataset/dataset_positive')
    positive_images = positive_loader.load_data()

    unlabeled_loader = CSVLoader('/srv/PU-dataset/unlabeled.csv', 'id', '/srv/PU-dataset/dataset_unlabeled')
    unlabeled_images = unlabeled_loader.load_data()

    extractor = ViTExtractor()
    #extractor = AutoencoderExtractor(input_shape=(224, 224, 3), filters=[8,16,32,64,64])
    positive_features, unlabeled_features = extractor.extract_features(positive_images, unlabeled_images)

    positive_features = positive_features.drop(['label', 'id'], axis=1).to_numpy()
    unlabeled_features = unlabeled_features.drop(['label', 'id'], axis=1).to_numpy()

    # Create partitions
    X_train, X_test, y_train, y_test = build_pu_data(
        positive_features, unlabeled_features,
        frac=0.1,
        known_positive_frac=0.5,
        positive_frac_in_unlabeled=0.5,
        test_split=0.2,
        test_split_positive=0.8
    )

    # Train the PU algorithm
    iterative_cls = IterativeClassifierAlgorithm(
        negative_detector=NaiveDetector(frac=0.1),
        stop_criterion=StopOnMetricDrop('f1'),
        classifier_class=svm.SVC,
        max_iterations=10
    )

    iterative_cls.fit(X_train, y_train, X_test, y_test)

    print(f'Evolution of f1 score: {iterative_cls.validation_results}')

if __name__ == "__main__":
    main()