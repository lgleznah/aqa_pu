import pandas as pd

def main():
    df = pd.read_csv('ava_ava_twostep_results.csv')

    df_exreport = df.pivot(
        index='extractor', 
        columns=['reliable_positive_threshold', 'classifier', 'detector'],
        values='balanced_accuracy'
    )

    df_exreport.columns = df_exreport.columns.to_flat_index()
    print(df_exreport)
    df_exreport.to_csv('exreport_input.csv')

if __name__ == "__main__":
    main()