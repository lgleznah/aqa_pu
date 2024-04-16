import sys
sys.path.append("../..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from pu.metrics import lift_curve_pu, aul_pu
from matplotlib.lines import Line2D

brief_cls_names = {
    'pt': 'ProbTagging',
    'nnpu': 'nnPU',
    'tsa': 'Two-step algorithm'
}

matplotlib.rcParams.update({'font.size': 15})

def get_xy_points(df: pd.DataFrame, exp_idx: int) -> tuple[list[float], list[float]]:
    y_pred = np.asarray(np.matrix(df[['y_pred']].values[exp_idx,0])).squeeze()
    y_true = np.asarray(np.matrix(df[['y_true_pu']].values[exp_idx,0])).squeeze()
    
    x_points, y_points = lift_curve_pu(y_true, y_pred)
    return x_points, y_points


def main():
    setting_names = ["ava_ava", "ava_aadb", "aadb_ava", "aadb_aadb"]
    for setting_name in setting_names:

        df = pd.read_csv(f'{setting_name}_results.csv')
        df['classifier'] = df['classifier'].map(brief_cls_names)
        df = df[df['percentile_threshold'] == 0.95]
        clasifiers = df["classifier"].unique()

        for idx, cls in enumerate(clasifiers):
            x_points, y_points = get_xy_points(df, idx)
            plt.plot(x_points, y_points, label=cls)

        plt.plot([0,1], [0,1], linestyle='--', c='black')
        plt.xlabel("$Y_{rate}$")
        plt.ylabel("$tpr$")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.title(f"{setting_name.split('_')[0].upper()} train, {setting_name.split('_')[1].upper()} test")
        plt.legend()
        plt.savefig(f'{setting_name}_liftcurve.pdf')

        plt.close()    

if __name__ == "__main__":
    main()