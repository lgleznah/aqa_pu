# This script iterates through all the images in a CSV, removing those with erros from such CSV
import sys
import os
import pandas as pd

from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm

'''
    Check a given CSV for corrupt images, and remove those images from the CSV.

    Parameters:
        - #1 CLI-argument: name of the CSV.
        - #2 CLI-argument: root folder with the images.
        - #3 CLI-argument: column with the image paths in the CSV.

    Returns:
        Nothing. Generates a new, error-free CSV
'''
def main():
    extractor = SentenceTransformer('clip-ViT-B-32')
    df = pd.read_csv(sys.argv[1])
    path_col = sys.argv[2]
    img_root = sys.argv[3]

    bad_idxs = []
    for i, img in tqdm(enumerate(df[[path_col]].squeeze())):
        try:
            path = os.path.join(img_root, img)
            _ = extractor.encode(Image.open(path))

        except KeyboardInterrupt:
            break

        except:
            print(f'Image {path} is corrupt!')
            bad_idxs.append(i)

    df_new = df.drop(index=bad_idxs)
    df_new.to_csv(os.path.basename(sys.argv[1]))

if __name__ == "__main__":
    main()