import sys

import seaborn as sns
import pandas as pd

def main():
    exp_name = sys.argv[1]
    title = sys.argv[2]
    baseline = sys.argv[3]
    baseline_file = sys.argv[4]

    df = pd.read_csv(f'{exp_name}_results.csv')
    df_good_extractor = df[df['extractor'] == 'clip-ViT-L-14']

    # Go from reliable positive threshold to percentiles


    plot = sns.lineplot(
        data = df_good_extractor,
        x = 'reliable_positive_threshold',
        y = 'balanced_accuracy',
        hue = 'classifier',
        style = 'detector',
        dashes = True,
        markers = True
    )

    plot.set_title(title)
    plot.set_ylim([0.4, 1.0])
    plot.set_xlabel("Reliable positive threshold")
    plot.set_ylabel("Balanced accuracy")

    # Load baselines
    df_baseline = pd.read_csv(f"{baseline_file}.csv")
    df_baseline_setting = df_baseline[df_baseline["setting"] == baseline]
    clasifiers = df_baseline["classifier"].unique()

    for idx, cls in enumerate(clasifiers):
        baseline_value = df_baseline_setting[df_baseline_setting["classifier"] == cls]["balanced_accuracy"].squeeze()
        plot.axhline(y = baseline_value, c = sns.color_palette("pastel")[idx], ls = (0, (3, 1, 1, 1, 1, 1)))

    plot.figure.savefig(f'{exp_name}_grahpic.pdf')

if __name__ == '__main__':
    main()