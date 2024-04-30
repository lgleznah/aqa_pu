import pandas as pd

def main() -> None:
    settings_names = ["ava_ava", "ava_aadb", "aadb_ava", "aadb_aadb"]
    metrics = ["balanced_accuracy", "accuracy", "f1", "aul"]
    full_df = pd.DataFrame()

    # Merge result dataframes into one
    for setting in settings_names:
        df = pd.read_csv(f"laion+{setting}_twostep_nomove_results.csv")
        df = df.drop(columns=["y_true", "y_pred", "Unnamed: 0", "reliable_positive_threshold", "detector", "extractor"])
        df = df.rename(columns={metric: f"{metric}_{setting}" for metric in metrics})
        full_df = pd.concat([full_df, df], axis=1)

    # Remove duplicate columns, split datasets per metric, and store these
    full_df = full_df.loc[:,~full_df.columns.duplicated()].copy()
    full_df = full_df.set_index("classifier")
    for metric in metrics:
        cols = [col for col in full_df.columns if col.startswith(metric)]

        metric_df = full_df[cols]
        metric_df.to_csv(f"{metric}_exreport.csv")

if __name__ == "__main__":
    main()