import pandas as pd

def main() -> None:
    setting_names = ["ava_ava", "ava_aadb", "aadb_ava", "aadb_aadb"]
    
    tsa_dfs, nnpu_dfs = [], []
    for setting_name in setting_names:
        df = pd.read_csv(f"{setting_name}_results.csv", index_col=False)
        df.columns = [col.replace("_", " ").capitalize() for col in df.columns]
        setting_name_col = setting_name.replace("_", "-").upper()

        # Generate tables for TSA and nnPU
        tsa_dfs.append(df[df["Classifier"] == "tsa"][["Percentile threshold", "Subclassifier", "Balanced accuracy", "Accuracy", "F1", "Aul"]])
        nnpu_dfs.append(df[df["Classifier"] == "nnpu"][["Percentile threshold", "Positive prior", "Balanced accuracy", "Accuracy", "F1", "Aul"]])

        tsa_dfs[-1]["Percentile threshold"] = tsa_dfs[-1]["Percentile threshold"].str.upper()
        nnpu_dfs[-1]["Percentile threshold"] = nnpu_dfs[-1]["Percentile threshold"].str.upper()

        tsa_dfs[-1]["Setting"] = setting_name_col
        nnpu_dfs[-1]["Setting"] = setting_name_col

    # Generate dataframes for LaTeX-ization
    tsa_df = pd.concat(tsa_dfs, axis=0)
    nnpu_df = pd.concat(nnpu_dfs, axis=0)

    for algo_df, column in zip([tsa_df, nnpu_df], ["Subclassifier", "Positive prior"]):
        for metric in ["Balanced accuracy", "Accuracy", "F1", "Aul"]:
            print(f"Column: {column}, metric: {metric}\n\n")
            print(
                algo_df.pivot(index=["Setting", column], columns="Percentile threshold", values=metric)
                    .style.format(precision=4)
                    .to_latex(hrules=True)
            )

if __name__ == "__main__":
    main()