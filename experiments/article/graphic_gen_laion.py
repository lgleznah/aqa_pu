import sys
sys.path.append("../..")

import pandas as pd
import matplotlib.pyplot as plt

from pu.metrics import aul_pu

def get_metric_or_aul(metric_name: str, df: pd.DataFrame) -> pd.Series:
    if (metric_name != "aul"):
        return df[metric_name].item()
    else:
        return df.apply(lambda row: aul_pu(row["y_true_pu"], row["y_pred"]), axis=1).item()

def main() -> None:
    setting_names = [f"laion+{x}" for x in ["ava_ava", "ava_aadb", "aadb_ava", "aadb_aadb"]]
    
    for setting_name in setting_names:
        df = pd.read_csv(f"{setting_name}_results.csv")

        # Plot a figure for each metric
        metrics = ["balanced_accuracy", "accuracy", "f1", "aul"]
        for metric in metrics:

            # Plot a bar for each algorithm
            algorithms = ["pt", "nnpu", "tsa"]
            colors = ["#fe218b", "#fed700", "#21b0fe"]
            metric_values = []
            for algo in algorithms:

                df_to_plot = df[df["classifier"] == algo]
                value = get_metric_or_aul(metric, df_to_plot)
                metric_values.append(value)

            plt.bar(algorithms, metric_values, color=colors, label=algorithms)
            
            plt.ylim([0, 1])
            plt.legend()
            plt.ylabel(metric.capitalize().replace('_', ' '))
            plt.xlabel("Algorithm")
            plt.title(setting_name.capitalize().replace('_', '-'))
            plt.savefig(f"{setting_name}_{metric}.pdf")
            plt.close()

if __name__ == "__main__":
    main()