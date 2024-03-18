import pandas as pd

def merge_dataframes_and_process(paths, scenario_names):
    dfs = []

    for path, scenario in zip(paths, scenario_names):
        df = pd.read_csv(path)
        df.insert(0, "setting", scenario)
        dfs.append(df)

    full_df = pd.concat(dfs, axis=0)
    return process_results(full_df, "PU")

def process_results(df, scenario):
    classifiers = df["classifier"].unique()

    setting_template = "LAION+{}-{}"

    train_col, cls_col, scenario_col, fall_col = [], [], [], []

    for setting in [[("AVA","AADB"), ("AVA","AVA")],[("AADB","AVA"), ("AADB","AADB")]]:
        df_different_test = df[df["setting"] == setting_template.format(*setting[0])]
        df_same_test = df[df["setting"] == setting_template.format(*setting[1])]

        for cls in classifiers:
            different_test_balacc = df_different_test[df_different_test["classifier"] == cls]["balanced_accuracy"].squeeze()
            same_test_balacc = df_same_test[df_same_test["classifier"] == cls]["balanced_accuracy"].squeeze()
            fall_ratio = round((1 - (different_test_balacc / same_test_balacc)) * 100, 2)

            train_col.append(setting[0][0])
            cls_col.append(cls)
            scenario_col.append(scenario)
            fall_col.append(fall_ratio)

    df = pd.DataFrame.from_dict({
        'train_ds': train_col,
        'classifier': cls_col,
        'setting': scenario_col,
        'balac_fall': fall_col
    })

    return df

def main():
    df_ti = process_results(pd.read_csv("baselines_nomove.csv"), "TI")
    df_pi = process_results(pd.read_csv("baselines_nolabels.csv"), "PI")

    df_pu = merge_dataframes_and_process(
        [
            "laion+ava_ava_twostep_nomove_results.csv",
            "laion+ava_aadb_twostep_nomove_results.csv",
            "laion+aadb_ava_twostep_nomove_results.csv",
            "laion+aadb_aadb_twostep_nomove_results.csv",
        ],
        [
            "LAION+AVA-AVA",
            "LAION+AVA-AADB",
            "LAION+AADB-AVA",
            "LAION+AADB-AADB"
        ]
    )

    results_df = pd.concat([df_ti, df_pi, df_pu], axis=0)
    results_df.to_csv("balacc_changes.csv")

    return

if __name__ == "__main__":
    main()