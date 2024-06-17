import pandas as pd
import seaborn as sns

def main():
    df = pd.read_csv("laion+ava_ava_knn_selection_results.csv")

    df["detector"] = df["detector"].str.slice(start=4).astype(int)

    plot = sns.lineplot(data=df, x="detector", y="aul", style="extractor", markers=True)

    plot.set_ylim([0.69, 0.71])
    plot.set_xlabel("Value of $k$")
    plot.set_ylabel("AUL")
    plot.set_yticks([0.69, 0.7, 0.71])
    plot.set_xticks(df["detector"].unique())
    plot.get_legend().remove()

    fig = plot.get_figure()
    fig.savefig("knn_plot.pdf") 

if __name__ == "__main__":
    main()