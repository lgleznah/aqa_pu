import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pu.metrics import lift_curve_pu

def get_xy_points(df: pd.DataFrame, exp_idx: int) -> tuple[list[float], list[float]]:
    y_pred = df[['y_pred']].values[exp_idx,0]
    y_pred = np.matrix(y_pred)
    y_pred = np.asarray(y_pred.reshape((y_pred.shape[1] // 2, 2)))[:,1]
    y_true = np.asarray(np.matrix(df[['y_true']].values[0,0])).squeeze()
    
    x_points, y_points = lift_curve_pu(y_true, y_pred)
    return x_points, y_points


def main():
    extractor = 'clip-ViT-L-14'
    exp_name = sys.argv[1]
    title = sys.argv[2]
    baseline = sys.argv[3]
    baseline_file = sys.argv[4]
    true_baselines_file = 'baselines_nolabels'

    colors = ['#ff218c', '#ffd800', '#21b1ff']

    df = pd.read_csv(f'{exp_name}_results.csv')
    df_extractor = df[df['extractor'] == extractor].reset_index(drop=True)
    clasifiers = df_extractor["classifier"].unique()

    for idx, cls in enumerate(clasifiers):
        x_points, y_points = get_xy_points(df_extractor, idx)
        plt.plot(x_points, y_points, label=cls, c=colors[idx])

    # Load PN baselines
    df_baseline = pd.read_csv(f"{baseline_file}.csv")
    df_baseline_setting = df_baseline[df_baseline["setting"] == baseline]
    clasifiers = df_baseline["classifier"].unique()

    for idx, cls in enumerate(clasifiers):
        x_points, y_points = get_xy_points(df_baseline_setting, idx)
        plt.plot(x_points, y_points, label=f'Baseline {cls}', c=colors[idx], linestyle='--', alpha=0.5)

    # Load non-labeled PN baselines
    df_true_baseline = pd.read_csv(f"{true_baselines_file}.csv")
    df_true_baseline_setting = df_true_baseline[df_true_baseline["setting"] == baseline]
    clasifiers = df_true_baseline["classifier"].unique()

    for idx, cls in enumerate(clasifiers):
        x_points, y_points = get_xy_points(df_true_baseline_setting, idx)
        plt.plot(x_points, y_points, label=f'Baseline {cls}', c=colors[idx], linestyle='dashdot', alpha=0.5)


    plt.plot([0,1], [0,1], linestyle=':', c='gray')
    plt.xlabel("$Y_{rate}$")
    plt.ylabel("$tpr$")
    plt.title(title)
    plt.legend()
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.savefig(f'{exp_name}_liftcurve.pdf')
    

if __name__ == "__main__":
    main()