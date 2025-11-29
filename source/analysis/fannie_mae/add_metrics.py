import pandas as pd
import numpy as np
import json
from pathlib import Path

from source.lib.save_data import save_data

def main():
    with open('source/lib/config.json', 'r') as f:
        CONFIG = json.load(f)
    
    INDIR = Path('datastore/output/derived/fannie_mae')
    OUTDIR = Path('datastore/output/derived/fannie_mae')
    SEED = CONFIG['SEED']
    SAMPLE_SIZE = CONFIG['SAMPLE_SIZE']
    print("Reading data...")
    ddf = dd.read_parquet(INDIR / 'sflp_sample.parquet')


# def add_predicted_counterfactual_rate(df):
#     pass

# def add_current_ltv(df):
#     pass

def add_ind_refi_eligible(df):
    # High FICO, low LTV, never missed a payment
    mask = (df['fico'] > 680) & (df['ltv'] < 90)
    df['refi_eligible'] = mask.astype(int)
    return df



def add_adl_threshold(df):
    # Refinance when i - i_0 < optimal refinance threshold
    # transaction costs, discount rate, marginal tax rate, probability of move --> compute optimal refinance threshold
    
    # Mortgage rate volatility
    mortgage_rate_vol = (
        df
        .select(columns=['mortgage30us', 'period'])
        .drop_duplicates(subset=['period'])
        .sort_values(by=['period'])
        .assign(mortgage_rate_diff = lambda x: x['mortgage30us'] - x['mortgage30us'].shift(1))
        ['mortgage_rate_diff']
        .dropna()
        .std() * np.sqrt(12)
    )
    
    # Transaction costs: 1 percent of the unpaid balance, plus $2,000
    df['transaction_costs'] = 0.01 * df['upb_curr'] + 2000
    
    discount_rate = 0.05
    prob_move = 0.1
    marginal_tax_rate = 0.28
    
    pass

def add_ind_should_refi(df):
    pass

def add_savings_from_refi(df):
    pass

if __name__ == "__main__":
    main()


