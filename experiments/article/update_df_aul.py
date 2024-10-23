import sys
sys.path.append("../..")

import pandas as pd

from pu.metrics import aul_pu

def main() -> None:
    setting_names = ["ava_ava", "ava_aadb", "aadb_ava", "aadb_aadb"]
    
    for setting_name in setting_names:
        df = pd.read_csv(f"{setting_name}_results.csv")
        df_baseline = pd.read_csv(f"{setting_name}_baseline_results.csv")
        df["aul"] = df.apply(lambda row: aul_pu(row["y_true_pu"], row["y_pred"]), axis=1)
        df_baseline["aul"] = df_baseline.apply(lambda row: aul_pu(row["y_true_pu"], row["y_pred"]), axis=1)
        
        df.to_csv(f"{setting_name}_results.csv")
        df_baseline.to_csv(f"{setting_name}_baseline_results.csv")


if __name__ == "__main__":
    main()