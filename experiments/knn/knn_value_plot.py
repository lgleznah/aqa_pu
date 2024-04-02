import pandas as pd
import seaborn as sns

def main():
    df_large_k = pd.read_csv("laion+ava_ava_knn_selection_results.csv")
    df_small_k = pd.read_csv("laion+ava_ava_knn_selection_small_results.csv")

    negative_detectors_large, balaccs_large = df_large_k[["detector"]], df_large_k[["balanced_accuracy"]]
    negative_detectors_small, balaccs_small = df_small_k[["detector"]], df_small_k[["balanced_accuracy"]]

    negative_detectors = pd.concat([negative_detectors_small, negative_detectors_large]).squeeze().str.slice(start=4).astype(int)
    balaccs = pd.concat([balaccs_small, balaccs_large]).squeeze()

    plot = sns.lineplot(x=negative_detectors, y=balaccs)
    fig = plot.get_figure()
    fig.savefig("knn_plot.pdf") 

if __name__ == "__main__":
    main()