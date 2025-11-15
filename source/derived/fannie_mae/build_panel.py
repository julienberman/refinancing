import pandas as pd
import numpy as np
from pathlib import Path

from source.lib.save_data import save_data

def main():
    INDIR = Path('datastore/output/derived/fannie_mae')
    INDIR_CW = Path('datastore/raw/crosswalks')
    
    fannie_mae = pd.read_parquet(INDIR / 'sflp_clean')
    cw_state_county = pd.read_csv(INDIR_CW / 'state_county.csv')
    
    fannie_mae_with_fips = add_fips(fannie_mae, cw_state_county)
    fannie_mae_with_indicators = add_event_indicators(fannie_mae_with_fips)
    
    pass

def add_fips(df, cw_state_county):
    pass

def add_event_indicators(df):
    
    indicator_map = {
        1: 'exit_t1',
        3: 'exit_t3',
        6: 'exit_t6',
        12: 'exit_t12',
        24: 'exit_t24'
    }
    
    df_with_indicators = df.copy()
    for window, indicator in indicator_map.items():
        df_with_indicators[indicator] = (
            df_with_indicators.groupby('loan_id')['exit_code']
            .transform(lambda x: x.shift(-window+1).rolling(window=window, min_periods=1).apply(lambda y: y.notna().any(), raw=True))
            .fillna(0)
            .astype(int)
        )

    return df_with_indicators


if __name__ == '__main__':
    main()

