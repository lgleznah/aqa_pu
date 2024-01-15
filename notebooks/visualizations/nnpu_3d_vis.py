import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def main():
    input_file = "../quantile_threshold_vit_results_ava_nnpu.csv"

    df = pd.read_csv(input_file)

    for feature_extractor in ['clip-ViT-B-32', 'clip-ViT-B-16', 'clip-ViT-L-14']:
        for metric in ["balanced_accuracy","accuracy","f1"]:
            df_extractor = df[df["extractor"] == feature_extractor]

            num_quantiles = df_extractor["quantile"].nunique()
            num_priors = df_extractor["prior"].nunique()
            priors = df_extractor["prior"].unique()

            results_2d = np.reshape(df_extractor[metric], (num_quantiles, num_priors))
            ax = sns.heatmap(results_2d)
            ax.set_xlabel("Prior")
            ax.set_ylabel("Quantile")
            ax.set_xticklabels(priors)
            ax.set_yticklabels([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99], rotation=0)
            ax.set_title(f"{feature_extractor} {metric.replace('_', ' ')} results")

            plt.savefig(f"result_{feature_extractor}_{metric}.svg")
            plt.close()


if __name__ == "__main__":
    main()