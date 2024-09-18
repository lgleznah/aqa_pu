import sys
sys.path.append("../..")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pu.metrics import aul_pu
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import matplotlib
matplotlib.rcParams.update({'font.size': 12})

def get_metric_or_aul(metric_name: str, df: pd.DataFrame) -> pd.Series:
    if (metric_name != "aul"):
        return df[metric_name]
    else:
        return df.apply(lambda row: aul_pu(row["y_true_pu"], row["y_pred"]), axis=1)
    
def generate_colors_from_palette(palette_name: str) -> list[tuple[float, float, float]]:
    colors = sns.color_palette(palette_name)
    old_colors = colors[2:5]
    colors[2] = colors[1]
    colors[1] = (1.0, 0.5058823529411765, 0.3098039215686274)
    colors[3] = (1.0, 0.9058823529411765, 0.7098039215686274)
    colors[4:7] = old_colors

    return colors
    
def plot_tsa_lines(df: pd.DataFrame, df_baseline: pd.DataFrame, metric: str, setting_name: str) -> None:
    subclassifiers = ["logistic", "knn-5", "knn-9", "knn-19", "nb", "rf", "svm"]
    
    colors_pu = generate_colors_from_palette("bright")
    colors_pn = generate_colors_from_palette("pastel")
    colors_base = generate_colors_from_palette("muted")
    fig, ax = plt.subplots()

    legend_handles = []

    for algo, color_pu, color_pn, color_base in zip(subclassifiers, colors_pu, colors_pn, colors_base):

        df_to_plot = df[(df["classifier"] == "tsa") & (df["subclassifier"] == algo) & (df["percentile_threshold"] != "pn")]
        xticklabels = df_to_plot["percentile_threshold"]
        x = list(range(len(xticklabels)))
        y = get_metric_or_aul(metric, df_to_plot)
        df.loc[df["classifier"] == algo, metric] = y

        ax.plot(x, y, color=color_pu, marker="+", linestyle='solid', label=f'Two-step algorithm ({algo})')
        
        # Plot baseline PN to PU stats
        df_baseline_to_plot = df_baseline[(df_baseline["classifier"] == algo) & (df_baseline["percentile_threshold"] != 'pn')]
        #x = df_baseline_to_plot["percentile_threshold"].astype(float)
        y = get_metric_or_aul(metric, df_baseline_to_plot)
        ax.plot(x, y, color=color_pn, marker="+", linestyle='dotted', label=f'U $\\rightarrow$ N baseline ({algo})')


        # Plot classic PN baseline stats
        df_baseline_to_plot_pn = df_baseline[(df_baseline["classifier"] == algo) & (df_baseline["percentile_threshold"] == 'pn')]
        y = get_metric_or_aul(metric, df_baseline_to_plot_pn).squeeze()
        ax.axhline(y, color=color_base, linestyle='--', label='Classic PN baseline')

        # Fill legend handles and labels
        legend_handle = Patch(color=color_pu, label=algo)
        legend_handles.append(legend_handle)
    
    if metric == "balanced_accuracy":
        ax.axhline(0.5, color='gray', linestyle="--")

    # Setup legend
    legend_algorithms = ax.legend(handles=legend_handles, bbox_to_anchor=(0., -0.3, 1., .102), loc='lower left', ncols=7, mode="expand", borderaxespad=0., prop={'size': 8})
    ax.add_artist(legend_algorithms)
    
    scenarios_handles = []
    scenarios_handles.append(Line2D([], [], color='black', linestyle='solid', label='PU (TSA)'))
    scenarios_handles.append(Line2D([], [], color='black', linestyle='dotted', label='PN (U $\\rightarrow$ N)'))
    scenarios_handles.append(Line2D([], [], color='black', linestyle='--', label='PN (Original labels)'))
    legend_scenarios = ax.legend(handles=scenarios_handles, bbox_to_anchor=(0., -0.37, 1., .102), loc='lower left', ncols=3, mode="expand", borderaxespad=0., prop={'size': 8})

    # Setup figure
    plt.ylim([0, 1])
    plt.xlim([0, 9])
    plt.ylabel(metric.capitalize().replace('_', ' '), fontdict={'fontsize': 15})
    plt.xlabel("Score percentile threshold", fontdict={'fontsize': 15})
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    plt.title(f"{setting_name.split('_')[0].upper()} train, {setting_name.split('_')[1].upper()} test", fontdict={'fontsize': 20})
    plt.savefig(f"{setting_name}_tsa_{metric}.pdf", bbox_extra_artists=(legend_algorithms,legend_scenarios), bbox_inches="tight")
    plt.close()

def plot_nnpu_heatmap(df: pd.DataFrame, metric: str, setting_name: str) -> None:
    df_to_plot = df[(df["classifier"] == "nnpu") & (df["percentile_threshold"] != "pn")]
    df.loc[df["classifier"] == "nnpu", metric] = get_metric_or_aul(metric, df_to_plot)
    df_to_plot = df_to_plot.pivot(index="percentile_threshold", columns="positive_prior", values=metric)

    ax = sns.heatmap(df_to_plot, annot=True, fmt=".2f", linewidth=.5, vmin=0, vmax=1, cmap='cubehelix')

    ax.set_xlabel(ax.xaxis.get_label().get_text().replace('_', ' ').capitalize())
    ax.set_ylabel(ax.yaxis.get_label().get_text().replace('_', ' ').capitalize())
    plt.title(f"{setting_name.split('_')[0].upper()} train, {setting_name.split('_')[1].upper()} test", fontdict={'fontsize': 20})
    plt.savefig(f"{setting_name}_nnpu_{metric}.pdf")
    plt.close()


def main() -> None:
    setting_names = ["ava_ava", "ava_aadb", "aadb_ava", "aadb_aadb"]
    
    for setting_name in setting_names:
        df = pd.read_csv(f"{setting_name}_results.csv")
        df_baseline = pd.read_csv(f"{setting_name}_baseline_results.csv")

        # Plot a figure for each metric
        metrics = ["balanced_accuracy", "accuracy", "f1"]
        for metric in metrics:

            # Plot each algorithm (lines for TSA, heatmap for NNPU)
            plot_tsa_lines(df, df_baseline, metric, setting_name)
            plot_nnpu_heatmap(df, metric, setting_name)        


if __name__ == "__main__":
    main()