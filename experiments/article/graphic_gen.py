import sys
sys.path.append("../..")

import pandas as pd
import matplotlib.pyplot as plt

from pu.metrics import aul_pu

def get_metric_or_aul(metric_name: str, df: pd.DataFrame) -> pd.Series:
    if (metric_name != "aul"):
        return df[metric_name]
    else:
        return df.apply(lambda row: aul_pu(row["y_true_pu"], row["y_pred"]), axis=1)

def main() -> None:
    setting_names = ["ava_ava", "ava_aadb", "aadb_ava", "aadb_aadb"]
    
    for setting_name in setting_names:
        df = pd.read_csv(f"{setting_name}_results.csv")

        # Plot a figure for each metric
        metrics = ["balanced_accuracy", "accuracy", "f1", "aul"]
        for metric in metrics:

            # Plot a line for each algorithm
            algorithms = ["pt", "nnpu", "tsa"]
            colors = ["#fe218b", "#fed700", "#21b0fe"]
            for algo, color in zip(algorithms, colors):

                df_to_plot = df[df["classifier"] == algo]
                x = df_to_plot["percentile_threshold"]
                y = get_metric_or_aul(metric, df_to_plot)

                plt.plot(x, y, color=color, marker="+", label=algo)
            
            plt.ylim([0, 1])
            plt.legend()
            plt.ylabel(metric.capitalize().replace('_', ' '))
            plt.xlabel("Score percentile thresold")
            plt.savefig(f"{setting_name}_{metric}.pdf")
            plt.close()

if __name__ == "__main__":
    main()