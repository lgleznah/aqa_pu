import sys
import pandas as pd

def main():
    df = pd.read_csv(f"{sys.argv[1]}_results_full.csv")
    results_latex = (
        df.pivot(index="classifier", columns="percentile_threshold", values="aul")
            .style.format(precision=4)
            .to_latex()
        )
    
    print(results_latex)

if __name__ == "__main__":
    main()