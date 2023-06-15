from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from PIL import Image
except:
    pass

try:
    import tensorflow as tf
    from tensorflow.keras import layers
    import sys
    sys.path.append('../datasets')
    from data.dataset_gen import paths_to_dataset
except:
    pass

from tqdm import tqdm

def chunkify(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Extractor(ABC):

    def __init__(self) -> None:
        self.project_root = os.environ['AQA_PU_ROOT']

    def extract_features(self, positive_images, unlabeled_images):
        features_file = os.path.join(self.project_root, 'embeddings', f'{self.filename}_{self.nfeatures}f.csv')

        if os.path.exists(features_file):
            features = pd.read_pickle(features_file, compression='gzip')
            positive_features = features[features['label'] == 'P']
            unlabeled_features = features[features['label'] == 'U']

            return (positive_features, unlabeled_features)

        else:
            positive_features, unlabeled_features = self._extract_features(positive_images, unlabeled_images)
            all_features = np.concatenate([positive_features, unlabeled_features])
            df = pd.DataFrame.from_dict({
                'id': np.concatenate([positive_images, unlabeled_images]),
                **{f'f{i}': all_features[:,i] for i, _ in enumerate(all_features.T)},
                'label': (['P'] * len(positive_images)) + (['U'] * len(unlabeled_images))
                })
            
            df.to_pickle(features_file, compression='gzip')
            return (positive_features, unlabeled_features)

    @abstractmethod
    def _extract_features(self, positive_images, unlabeled_images):
        pass


class AutoencoderExtractor(Extractor):

    def __init__(self, input_shape, filters, epochs=10) -> None:
        super().__init__()

        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()
        self.epochs = epochs
        self.input_shape = input_shape
        self.encoder.add(layers.Input(shape=input_shape))
        for filter in filters:
            self.encoder.add(layers.Conv2D(filter, (3,3), activation='relu', padding='same', strides=2))

        for filter in filters[::-1]:
            self.decoder.add(layers.Conv2DTranspose(filter, kernel_size=3, strides=2, activation='relu', padding='same'))

        self.decoder.add(layers.Conv2D(input_shape[2], kernel_size=(3,3), activation='sigmoid', padding='same'))

        self.autoencoder = tf.keras.Sequential([self.encoder, self.decoder])
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        self.filename = f'autoencoder_input_{"_".join([str(x) for x in input_shape])}_filters_{"_".join([str(x) for x in filters])}_epochs_{epochs}'
        self.nfeatures = self.encoder.layers[-1].output_shape[-1] * self.encoder.layers[-1].output_shape[-2] * self.encoder.layers[-1].output_shape[-3]


    def _extract_features(self, positive_images, unlabeled_images):
        positive_dataset = paths_to_dataset(positive_images, self.input_shape[:2])
        unlabeled_dataset = paths_to_dataset(unlabeled_images, self.input_shape[:2])
        full_dataset = positive_dataset.concatenate(unlabeled_dataset)

        self.autoencoder.fit(full_dataset, epochs=self.epochs)

        positive_features = self.encoder.predict(positive_dataset)
        unlabeled_features = self.encoder.predict(unlabeled_dataset)

        return (
            tf.reshape(positive_features, [len(positive_images), -1]), 
            tf.reshape(unlabeled_features, [len(unlabeled_images), -1])
        )
    

class ViTExtractor(Extractor):

    def __init__(self, extractor_name='clip-ViT-B-32') -> None:
        super().__init__()

        self.extractor = SentenceTransformer(extractor_name)
        self.filename = f'vit_name_{extractor_name}'
        self.nfeatures = self.extractor.encode('john madden aeiou').shape[0]

    def _extract_features(self, positive_images, unlabeled_images):

        chunks_positive = list(chunkify(positive_images, 900))

        positive_features = []
        for chunk in tqdm(chunks_positive):
            images = [Image.open(img) for img in chunk]
            positive_features.extend(self.extractor.encode(images, batch_size=32))

        chunks_unlabeled = list(chunkify(unlabeled_images, 900))

        unlabeled_features = []
        for chunk in tqdm(chunks_unlabeled):
            images = [Image.open(img) for img in chunk]
            unlabeled_features.extend(self.extractor.encode(images, batch_size=32))

        
        return (positive_features, unlabeled_features)