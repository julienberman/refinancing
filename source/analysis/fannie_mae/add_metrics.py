import pandas as pd
import numpy as np
import json
from pathlib import Path

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from source.lib.save_data import save_data

with open('source/analysis/fannie_mae/globals.json', 'r') as f:
    GLOBALS = json.load(f)

ANNUAL_DISCOUNT_RATE = GLOBALS["ANNUAL_DISCOUNT_RATE"]
PROB_MOVE = GLOBALS["PROB_MOVE"]
MARGINAL_TAX_RATE = GLOBALS["MARGINAL_TAX_RATE"]
EXPECTED_INFLATION = GLOBALS["EXPECTED_INFLATION"]

def main():
    with open('source/lib/config.json', 'r') as f:
        CONFIG = json.load(f)
    
    INDIR  = Path('datastore/output/derived/fannie_mae')
    OUTDIR = Path('datastore/output/derived/fannie_mae')

    print("Reading data...")
    ddf = dd.read_parquet(INDIR / 'sflp_sample.parquet')

    save_data(savings, OUTDIR / "savings_estimates.parquet")

def add_ind_refi_eligible(df):
    df["refi_eligible"] = ((df["fico"] > 680) & (df["ltv"] < 90)).astype(int)
    return df

def compute_monthly_payment(df):
    """Compute monthly mortgage payment"""
    r = (df["rate_orig"] / 100) / 12
    P = df["upb_orig"]
    T = df["term"]
    df["monthly_payment"] = P * (r * (1+r)**T) / ((1+r)**T - 1)
    return df

def add_adl_threshold(df):
    """
    Compute the ADL optimal refinance thresholds.
    """
    mortgage_rate_vol = compute_mortgage_rate_vol(df)
    k = 0.01 * df["upb_curr"] + 2000
    annual_payment = 12 * df["monthly_payment"]
    lam = PROB_MOVE + (annual_payment / df["upb_curr"] - (df["rate_orig"] / 100)) + EXPECTED_INFLATION
    df["adl_threshold"] = 100 * np.sqrt((mortgage_rate_vol * k) / (df["upb_curr"] * (1 - MARGINAL_TAX_RATE))) * np.sqrt(2 * (ANNUAL_DISCOUNT_RATE + lam))
    df["should_refi"] = (df["rate_gap"] > df["adl_threshold"]).astype(int)
    return df

def compute_mortgage_rate_vol(df):
    mortgage_rate_vol = (
        df[["mortgage30us", "period"]]
        .drop_duplicates("period")
        .sort_values("period")
        .assign(dr=lambda x: x["mortgage30us"].diff() / 100)
        ["dr"]
        .dropna()
        .std() * np.sqrt(12)
    )
    return mortgage_rate_vol

def add_savings_from_refi(group):    
    group['npv_no_refi'] = compute_npv_no_refi(group)
    group['npv_opt_refi'] = compute_npv_optimal_refi(group)
    group['npv_real_refi'] = compute_npv_real_refi(group)
    
    group['savings_opt'] =  group['npv_no_refi'] - group['npv_opt_refi']
    group['savings_real'] = group['npv_no_refi'] - group['npv_real_refi']
    group['savings_loss_from_suboptimal'] = (group['npv_no_refi'] - group['npv_opt_refi']) - (group['npv_no_refi'] - group['npv_real_refi'])
    return group

def compute_npv_no_refi(group):
    """Compute NPV of no refinance scenario."""
    term = group['term'].iloc[0]
    monthly_payment = group["monthly_payment"].iloc[0]
    
    d = ANNUAL_DISCOUNT_RATE / 12
    t = np.arange(1, term + 1)
    discount_factor = 1 / (1 + d)**t
    
    npv = monthly_payment * discount_factor.sum()
    return npv

def compute_npv_optimal_refi(group):
    """Compute NPV of optimal refinance scenario."""
    d = ANNUAL_DISCOUNT_RATE / 12
    
    old_monthly_payment = group["monthly_payment"].iloc[0]
    old_term = group['term'].iloc[0]
    old_t = np.arange(1, old_term + 1)
    old_discount_factor = 1 / (1 + d)**old_t
    
    # Get earliest date where should_refi is 1
    refi_mask = group['should_refi'] == 1
    if not refi_mask.any():
        npv = (old_monthly_payment * old_discount_factor).sum()
        return npv
    
    earliest_idx = group.loc[refi_mask, 'period'].idxmin()
    refi_period = group.loc[earliest_idx, 'period']
    new_rate = (group.loc[earliest_idx, 'rate_mortgage30us'] / 100) / 12
    new_term = group.loc[earliest_idx, 'time_to_maturity']
    new_upb = group.loc[earliest_idx, 'upb_curr']
    
    new_monthly_payment = new_upb * (new_rate * (1+new_rate)**new_term) / ((1+new_rate)**new_term - 1)
    
    t_before = np.arange(1, refi_period + 1)
    t_after = np.arange(refi_period + 1, old_term + 1)
    discount_factor_before = 1 / (1 + d)**t_before
    discount_factor_after = 1 / (1 + d)**t_after
    
    npv_before = old_monthly_payment * discount_factor_before.sum()
    npv_after = new_monthly_payment * discount_factor_after.sum()

    npv = npv_before + npv_after
    return npv

def compute_npv_real_refi(group):
    """Compute NPV of real refinance scenario."""
    d = ANNUAL_DISCOUNT_RATE / 12
    
    old_monthly_payment = group["monthly_payment"].iloc[0]
    old_term = group['term'].iloc[0]
    old_t = np.arange(1, old_term + 1)
    old_discount_factor = 1 / (1 + d)**old_t
    
    # Get loan exit period
    exit_period = group['exit_period'].iloc[0]
    if pd.isna(exit_period):
        npv = (old_monthly_payment * old_discount_factor).sum()
        return npv
    
    refi_period = int(exit_period - 1)
    new_rate = (group.loc[group['period'] == refi_period, 'rate_mortgage30us'].iloc[0] / 100) / 12
    new_term = group.loc[group['period'] == refi_period, 'time_to_maturity'].iloc[0]
    new_upb = group.loc[group['period'] == refi_period, 'upb_curr'].iloc[0]
    
    new_monthly_payment = new_upb * (new_rate * (1+new_rate)**new_term) / ((1+new_rate)**new_term - 1)
    
    t_before = np.arange(1, refi_period + 1)
    t_after = np.arange(refi_period + 1, old_term + 1)
    discount_factor_before = 1 / (1 + d)**t_before
    discount_factor_after = 1 / (1 + d)**t_after
    
    npv_before = old_monthly_payment * discount_factor_before.sum()
    npv_after = new_monthly_payment * discount_factor_after.sum()

    npv = npv_before + npv_after
    return npv

if __name__ == "__main__":
    main()

