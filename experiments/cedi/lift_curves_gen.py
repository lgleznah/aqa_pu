import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from pu.metrics import lift_curve_pu, aul_pu
from matplotlib.lines import Line2D

brief_cls_names = {
    'logistic': 'Logistic',
    'kneighbours-5': 'kNN-5',
    'kneighbours-9': 'kNN-9',
    'kneighbours-19': 'kNN-19',
    'naivebayes': 'NB',
    'rf': 'RF',
    'svm': 'SVM'
}

matplotlib.rcParams.update({'font.size': 15})

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

    colors = sns.color_palette("bright")
    old_colors = colors[2:5]
    colors[2] = colors[1]
    colors[1] = (1.0, 0.25, 0.0)
    colors[3] = (1.0, 0.75, 0.0)
    colors[4:7] = old_colors

    df = pd.read_csv(f'{exp_name}_results.csv')
    df['classifier'] = df['classifier'].map(brief_cls_names)
    df_extractor = df[df['extractor'] == extractor].reset_index(drop=True)
    clasifiers = df_extractor["classifier"].unique()

    for idx, cls in enumerate(clasifiers):
        x_points, y_points = get_xy_points(df_extractor, idx)
        plt.plot(x_points, y_points, label=cls, c=colors[idx])

    # Load PN baselines
    '''
    df_baseline = pd.read_csv(f"{baseline_file}.csv")
    df_baseline_setting = df_baseline[df_baseline["setting"] == baseline]
    clasifiers = df_baseline["classifier"].unique()

    for idx, cls in enumerate(clasifiers):
        x_points, y_points = get_xy_points(df_baseline_setting, idx)
        plt.plot(x_points, y_points, label=f'Baseline {cls}', c=colors[idx], linestyle='dotted', alpha=0.5)
    '''
        
    # Load non-labeled PN baselines
    df_true_baseline = pd.read_csv(f"{true_baselines_file}.csv")
    df_true_baseline['classifier'] = df_true_baseline['classifier'].map(brief_cls_names)
    df_true_baseline_setting = df_true_baseline[df_true_baseline["setting"] == baseline]
    clasifiers = df_true_baseline["classifier"].unique()

    for idx, cls in enumerate(clasifiers):
        x_points, y_points = get_xy_points(df_true_baseline_setting, idx)
        plt.plot(x_points, y_points, label=f'Baseline {cls}', c=colors[idx], linestyle='dotted')


    plt.plot([0,1], [0,1], linestyle='--', c='black')
    plt.xlabel("$Y_{rate}$")
    plt.ylabel("$tpr$")
    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(f'{exp_name}_liftcurve.pdf')

    plt.close()

    fig = plt.figure(4, figsize=(6,2))
    ax_legend = fig.add_subplot(111)
    ax_legend.set_axis_off()

    # Add algorithms to legend
    algo_legend_lines = []
    for idx, cls in enumerate(clasifiers):
        line = Line2D([0], [0], marker='o', 
                      color='w', 
                      label=cls, 
                      markerfacecolor=colors[idx], markersize=15)
        algo_legend_lines.append(line)

    # Add scenarios to legend
    scen_legend_lines = []
    scen_legend_lines.append(Line2D(
        [0], [0], color='black', label = 'PU'
    ))

    scen_legend_lines.append(Line2D(
        [0], [0], color='black', ls='dotted', label = 'Parcialmente informado'
    ))

    algo_legend = plt.legend(
        handles=algo_legend_lines, 
        title="Algoritmos", 
        loc='upper center', 
        framealpha=0.3, 
        ncols=7, 
        columnspacing=1.0, 
        markerscale=0.5,
        handletextpad=0
    )
    scen_legend = plt.legend(
        handles=scen_legend_lines, 
        title="Escenarios", 
        loc='center', 
        framealpha=0.3, 
        ncols=2, 
        columnspacing=1.0, 
        markerscale=0.5,
        handletextpad=0.3
    )

    ax_legend.add_artist(algo_legend)
    ax_legend.add_artist(scen_legend)
    plt.savefig("legend_lift.pdf")
    

if __name__ == "__main__":
    main()