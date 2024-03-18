import sys

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

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

matplotlib.rcParams.update({'font.size': 16})

def main():
    exp_name = sys.argv[1]
    title = sys.argv[2]
    baseline = sys.argv[3]
    baseline_file = sys.argv[4]
    true_baselines_file = 'baselines_nolabels'

    df = pd.read_csv(f'{exp_name}_results.csv')
    df_good_extractor = df[df['extractor'] == 'clip-ViT-L-14']
    df_good_extractor['classifier'] = df_good_extractor['classifier'].map(brief_cls_names)

    fig = plt.figure(4, figsize=(6,6))
    ax_fig = fig.add_subplot(111)

    colors_lines = sns.color_palette("bright")
    old_colors = colors_lines[2:5]
    colors_lines[2] = colors_lines[1]
    colors_lines[1] = (1.0, 0.25, 0.0)
    colors_lines[3] = (1.0, 0.75, 0.0)
    colors_lines[4:7] = old_colors

    colors_bars = sns.color_palette("pastel")
    old_colors = colors_bars[2:5]
    colors_bars[2] = colors_bars[1]
    colors_bars[1] = (1.0, 0.5058823529411765, 0.3098039215686274)
    colors_bars[3] = (1.0, 0.9058823529411765, 0.7098039215686274)
    colors_bars[4:7] = old_colors

    plot = sns.barplot(
        data = df_good_extractor,
        x = 'classifier',
        y = 'balanced_accuracy',
        palette=colors_bars[:7],
        legend=False,
        hue='classifier',
        ax=ax_fig
    )

    plot.set_title(title)
    plot.set_ylim([0.0, 0.8])
    plot.set_xlabel("Clasificador")
    plot.set_ylabel("Balanced accuracy")
    plot.tick_params(axis='x', labelrotation=35)

    # Load PN baselines
    df_baseline = pd.read_csv(f"{baseline_file}.csv")
    df_baseline['classifier'] = df_baseline['classifier'].map(brief_cls_names)
    df_baseline_setting = df_baseline[df_baseline["setting"] == baseline]
    clasifiers = df_baseline["classifier"].unique()

    for idx, cls in enumerate(clasifiers):
        baseline_value = df_baseline_setting[df_baseline_setting["classifier"] == cls]["balanced_accuracy"].squeeze()
        line = plot.axhline(y = baseline_value, c = colors_lines[idx], ls='dotted')

    # Load non-labeled PN baselines
    df_true_baseline = pd.read_csv(f"{true_baselines_file}.csv")
    df_true_baseline['classifier'] = df_true_baseline['classifier'].map(brief_cls_names)
    df_true_baseline_setting = df_true_baseline[df_true_baseline["setting"] == baseline]
    clasifiers = df_true_baseline["classifier"].unique()

    for idx, cls in enumerate(clasifiers):
        baseline_value = df_true_baseline_setting[df_true_baseline_setting["classifier"] == cls]["balanced_accuracy"].squeeze()
        line = plot.axhline(y = baseline_value, c = colors_lines[idx], ls = '--')

    # Add algorithms to legend
    algo_legend_lines = []
    for idx, cls in enumerate(clasifiers):
        line = Line2D([0], [0], marker='o', 
                      color='w', 
                      label=cls, 
                      markerfacecolor=colors_lines[idx], markersize=15)
        algo_legend_lines.append(line)

    # Add scenarios to legend
    scen_legend_lines = []
    scen_legend_lines.append(Line2D(
        [0], [0], color='black', ls='dotted', label = 'Totalmente informado'
    ))

    scen_legend_lines.append(Line2D(
        [0], [0], color='black', ls='--', label = 'Parcialmente informado'
    ))

    plt.tight_layout()
    plot.figure.savefig(f'{exp_name}_grahpic.pdf')

    plt.close()

    fig = plt.figure(4, figsize=(6,2))
    ax_legend = fig.add_subplot(111)
    ax_legend.set_axis_off()

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
    plt.savefig("legend.pdf")

if __name__ == '__main__':
    main()