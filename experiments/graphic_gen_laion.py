import sys

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    exp_name = sys.argv[1]
    title = sys.argv[2]
    baseline = sys.argv[3]
    baseline_file = sys.argv[4]
    true_baselines_file = 'baselines_nolabels'

    df = pd.read_csv(f'{exp_name}_results.csv')
    df_good_extractor = df[df['extractor'] == 'clip-ViT-L-14']

    plot = sns.barplot(
        data = df_good_extractor,
        x = 'classifier',
        y = 'balanced_accuracy',
        hue = 'detector'
    )

    plot.set_title(title)
    plot.set_ylim([0.0, 1.0])
    plot.set_xlabel("Classifier")
    plot.set_ylabel("Balanced accuracy")
    #cls_legend = plot.legend(title="Detector", loc='upper right', framealpha=0.3)

    # Load PN baselines
    df_baseline = pd.read_csv(f"{baseline_file}.csv")
    df_baseline_setting = df_baseline[df_baseline["setting"] == baseline]
    clasifiers = df_baseline["classifier"].unique()

    baseline_lines = []
    for idx, cls in enumerate(clasifiers):
        baseline_value = df_baseline_setting[df_baseline_setting["classifier"] == cls]["balanced_accuracy"].squeeze()
        line = plot.axhline(y = baseline_value, c = sns.color_palette("pastel")[idx], ls = (0, (3, 1, 1, 1, 1, 1)), label=cls)
        baseline_lines.append(line)

    # Load non-labeled PN baselines
    df_true_baseline = pd.read_csv(f"{true_baselines_file}.csv")
    df_true_baseline_setting = df_true_baseline[df_true_baseline["setting"] == baseline]
    clasifiers = df_true_baseline["classifier"].unique()

    true_baseline_lines = []
    for idx, cls in enumerate(clasifiers):
        baseline_value = df_true_baseline_setting[df_true_baseline_setting["classifier"] == cls]["balanced_accuracy"].squeeze()
        line = plot.axhline(y = baseline_value, c = sns.color_palette("tab10")[idx], ls = '--', label=cls)
        true_baseline_lines.append(line)

    baseline_legend = plt.legend(baseline_lines, clasifiers, title="Baselines", loc='upper left', framealpha=0.3)
    true_baseline_legend = plt.legend(true_baseline_lines, clasifiers, title="Baselines w/o labels", loc='upper right', framealpha=0.3)
    plot.add_artist(baseline_legend)
    plot.add_artist(true_baseline_legend)
    #plot.add_artist(cls_legend)
    plot.figure.savefig(f'{exp_name}_grahpic.pdf')

if __name__ == '__main__':
    main()