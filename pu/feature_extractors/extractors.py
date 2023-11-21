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
    from pu.data.dataset_gen import paths_to_dataset
except:
    pass

from tqdm import tqdm

def chunkify(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Extractor(ABC):
    '''
    Feature extraction algorithms

    This is the superclass of all feature extraction algorithms. These take an image dataset as input,
    and return lower-dimensional features which serve as the input of the PU algorithms.

    Parameters
    ----------
    experiment_name: name of the experiment (to use the same parameters for multiple datasets)
    '''

    def __init__(self, experiment_name) -> None:
        self.project_root = os.environ['AQA_PU_ROOT']
        self.experiment_name = experiment_name

    def extract_features(self, positive_images, unlabeled_images, use_cache=True):
        '''
        Extract features from images

        This method acts as a wrapper for _extract_features, which is the method that extracts features
        from the images. This wrapper offers checkpointing functionality, avoiding having to run
        the actual feature extraction algorithm twice.

        Parameters
        ----------
        positive_images: list of paths to positive images
        unlabeled_images: list of paths to unlabeled images
        use_cache: whether to reuse or save the features from or into a cache file
        '''

        features_file = os.path.join(self.project_root, 'embeddings', f'{self.experiment_name}_{self.filename}_{self.nfeatures}f.csv')

        if use_cache and os.path.exists(features_file):
            features = pd.read_pickle(features_file, compression='gzip')
            positive_features = features[features['label'] == 1]
            unlabeled_features = features[features['label'] == 0]

            return (positive_features, unlabeled_features)

        else:
            positive_features, unlabeled_features = self._extract_features(positive_images, unlabeled_images)
            all_features = np.concatenate([positive_features, unlabeled_features])
            df = pd.DataFrame.from_dict({
                'id': np.concatenate([positive_images, unlabeled_images]),
                **{f'f{i}': all_features[:,i] for i, _ in enumerate(all_features.T)},
                'label': ([1] * len(positive_images)) + ([0] * len(unlabeled_images))
                })
            
            if use_cache:
                df.to_pickle(features_file, compression='gzip')
                
            return (df[df['label'] == 1], df[df['label'] == 0])

    @abstractmethod
    def _extract_features(self, positive_images, unlabeled_images):
        pass


class AutoencoderExtractor(Extractor):
    '''
    Autoencoder-based feature extraction

    This feature extractor trains an autoencoder on the positive and unlabeled images,
    keeping the outputs of the encoder as features.

    Parameters
    ----------
    experiment_name: name of the experiment (to use the same parameters for multiple datasets)
    input_shape: input image size
    filters: size of the convolutional filters to employ
    epochs: for how many epochs the autoencoder should be trained
    '''

    def __init__(self, experiment_name, input_shape, filters, epochs=10) -> None:
        super().__init__(experiment_name)

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
        '''
        Use the autoencoder to extract features

        Train the autoencoder, and return the outputs of the encoder
        '''
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
    '''
    ViT-based feature extractor

    This feature extractor employs a pretrained vision transformer to extract the
    features of the images, using its output as features.

    Parameters
    ----------
    experiment_name: name of the experiment (to use the same parameters for multiple datasets)
    extractor_name: name of the ViT to employ, as specified by the SentenceTransformers library
    '''

    def __init__(self, experiment_name, extractor_name='clip-ViT-B-32') -> None:
        super().__init__(experiment_name)

        self.extractor = SentenceTransformer(extractor_name)
        self.filename = f'vit_name_{extractor_name}'
        self.nfeatures = self.extractor.encode('john madden aeiou').shape[0]

    def _extract_features(self, positive_images, unlabeled_images):
        '''
        Use the ViT to extract features
        '''

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