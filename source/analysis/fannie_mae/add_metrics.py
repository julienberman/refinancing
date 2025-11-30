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
        / 100
        .dropna()
        .std() * np.sqrt(12)
    )
    
    rate_orig = (df['rate_orig'] / 100)
    upb_curr = df['upb_curr']
    upb_orig = df['upb_orig']
    term = df['term']
    expected_inflation = 0.025
    discount_rate = 0.05
    prob_move = 0.1
    marginal_tax_rate = 0.28
    
    # Transaction costs: 1 percent of the unpaid balance, plus $2,000
    k = 0.01 * df['upb_curr'] + 2000
    
    # Annual p&i repayment
    p = 12 * (upb_orig * (rate_orig / 12)) / (1 - (1 + (rate_orig / 12))**(-term))
    
    lbda = prob_move + (p / upb_curr - rate_orig) + expected_inflation
    
    df['adl_threshold'] = 100 * np.sqrt( ((mortgage_rate_vol * k) / (upb_curr * (1 - marginal_tax_rate))) * np.sqrt(2*(discount_rate + lbda)) )
    df['should_refi'] = (df['rate_gap'] > df['adl_threshold']).astype(int)
    
    return df

def add_savings_from_refi(df):
    pass

if __name__ == "__main__":
    main()


