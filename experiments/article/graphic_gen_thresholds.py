import sys
sys.path.append("../..")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pu.metrics import aul_pu

matplotlib.rcParams.update({'font.size': 12})

def get_metric_or_aul(metric_name: str, df: pd.DataFrame) -> pd.Series:
    if (metric_name != "aul"):
        return df[metric_name]
    else:
        return df.apply(lambda row: aul_pu(row["y_true_pu"], row["y_pred"]), axis=1)

def main() -> None:
    setting_names = ["ava_ava", "ava_aadb", "aadb_ava", "aadb_aadb"]
    
    for setting_name in setting_names:
        df = pd.read_csv(f"{setting_name}_results.csv")
        df_baseline = pd.read_csv(f"{setting_name}_baseline_results.csv")

        # Plot a figure for each metric
        metrics = ["balanced_accuracy", "accuracy", "f1", "aul"]
        for metric in metrics:

            # Plot a line for each algorithm
            algorithms = ["pt", "nnpu", "tsa"]
            colors = ["#fe218b", "#fed700", "#21b0fe", "#22cb00"]
            for algo, color in zip(algorithms, colors):

                df_to_plot = df[df["classifier"] == algo]
                x = df_to_plot["percentile_threshold"]
                y = get_metric_or_aul(metric, df_to_plot)
                df.loc[df["classifier"] == algo, metric] = y

                plt.plot(x, y, color=color, marker="+", label=algo)

            # Plot baseline stats
            df_baseline_to_plot = df_baseline[(df_baseline["classifier"] == 'logistic') & (df_baseline["percentile_threshold"] != 'pn')]
            x = df_baseline_to_plot["percentile_threshold"].astype(float)
            y = get_metric_or_aul(metric, df_baseline_to_plot)

            plt.plot(x, y, color=colors[3], marker="+", label='U $\\rightarrow$ N baseline')

            # Plot classic PN baseline stats
            df_baseline_to_plot_pn = df_baseline[(df_baseline["classifier"] == 'logistic') & (df_baseline["percentile_threshold"] == 'pn')]
            x = df_baseline_to_plot_pn["percentile_threshold"]
            y = get_metric_or_aul(metric, df_baseline_to_plot_pn).squeeze()

            plt.axhline(y, color=colors[3], linestyle='--', label='Classic PN baseline')
            
            if metric == "balanced_accuracy":
                plt.axhline(0.5, color='gray', linestyle="--")

            plt.ylim([0, 1])
            plt.legend()
            plt.ylabel(metric.capitalize().replace('_', ' '), fontdict={'fontsize': 15})
            plt.xlabel("Score percentile thresold", fontdict={'fontsize': 15})
            plt.title(f"{setting_name.split('_')[0].upper()} train, {setting_name.split('_')[1].upper()} test", fontdict={'fontsize': 20})
            plt.savefig(f"{setting_name}_{metric}.pdf")
            plt.close()

        df = df.drop(columns=["y_pred", "y_true", "y_true_pu"])
        df.to_csv(f"{setting_name}_results_full.csv")

if __name__ == "__main__":
    main()