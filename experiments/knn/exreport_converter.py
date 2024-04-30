import sys
sys.path.append("../..")

import pandas as pd

from pu.metrics import aul_pu

def main():
    df = pd.read_csv('laion+ava_ava_knn_selection_results.csv')

    df_exreport = df.pivot(
        index='detector', 
        columns=['seed'],
        values='aul'
    )

    df_exreport.to_csv('exreport_input.csv')

if __name__ == "__main__":
    main()