from feature_extractors.extractors import ViTExtractor, AutoencoderExtractor
from data.loaders import CSVLoader
from data.pu_builder import build_pu_data

import os

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
    unlabeled_features= unlabeled_features.drop(['label', 'id'], axis=1).to_numpy()

    # Create partitions
    X_train, X_test, y_train, y_test = build_pu_data(
        positive_features, unlabeled_features,
        frac=0.1,
        known_positive_frac=0.5,
        positive_frac_in_unlabeled=0.5,
        test_split=0.2
    )

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

if __name__ == "__main__":
    main()