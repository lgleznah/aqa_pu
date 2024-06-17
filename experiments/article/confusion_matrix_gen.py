import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(experiment_name: str) -> None:
    df = pd.read_csv(f'{experiment_name}_results.csv')

    for _, row in df.iterrows():
        threshold = row['percentile_threshold']
        cls = row['classifier']
        confmat = np.fromstring(row['confusion_matrix'].replace("[", "").replace("]", ""), sep=',').reshape((2,2)).astype(int)

        ax = sns.heatmap(confmat, annot=True, fmt='g')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")

        fig = ax.get_figure()
        fig.savefig(f'confmats/{experiment_name}_{str(threshold)}_{cls}.eps')
        plt.close()

def main() -> None:
    experiments = ['ava_ava', 'ava_aadb', 'aadb_ava', 'aadb_aadb', 
                   'laion+ava_ava', 'laion+ava_aadb', 'laion+aadb_ava', 'laion+aadb_aadb'
                    ]
    
    for experiment in experiments:
        plot_matrix(experiment)

if __name__ == '__main__':
    main()