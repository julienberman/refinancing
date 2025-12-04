import pandas as pd
from pathlib import Path

from source.lib.JMSLab.autofill import GenerateAutofillMacros

def main():
    INDIR = Path('datastore/output/derived/fannie_mae')
    OUTDIR = Path('output/analysis/figure_refi_delay')

    df = pd.read_parquet(INDIR / 'sflp_sample_processed_high.parquet')
    
    df_agg = (
        df
        .groupby('loan_id', as_index=False)
        .agg({'should_refi': longest_run_of_ones,})
    )
    mean_refi_delay = df_agg['should_refi'].mean()
    
    GenerateAutofillMacros(
        ["mean_refi_delay"],
        "{:,.0f}",
        OUTDIR / "autofill_mean_refi_delay.tex"
    )

def longest_run_of_ones(x):
    runs = (x != x.shift()).cumsum()
    return (x.groupby(runs).cumsum() * x).max()

if __name__ == '__main__':
    main()

