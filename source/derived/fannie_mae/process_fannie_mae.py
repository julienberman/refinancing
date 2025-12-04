import datetime
import numpy as np
import pandas as pd
import janitor
import json
import glob
import pyarrow
from pathlib import Path

from source.lib.save_data import save_data

def main():
    with open('source/derived/fannie_mae/parameters.json', 'r') as f:
        PARAMETER_LIST = json.load(f)
    
    INDIR_SFLP = Path('datastore/output/derived/fannie_mae')
    INDIR_FRED = Path('output/derived/fred')
    INDIR_CW = Path('datastore/raw/crosswalks/data')
    OUTDIR = Path('datastore/output/derived/fannie_mae')
    MASK_FULL_SAMPLE = (df["exit_code"] == "prepaid") & (df['mortgage_type'] == 'fixed') & (df["time_to_exit"] >= 1) & (df["time_from_orig"] >= 0) 
    MASK_REFI_ELIGIBLE = MASK_FULL_SAMPLE & (df["credit_score_orig"] > 680) & (df["ltv"] < 90) & (df["dlq_status"] == 0)

    df = pd.read_parquet(INDIR_SFLP / 'sflp_clean_sample.parquet')
    mortgage30us = pd.read_csv(INDIR_FRED / 'mortgage30us.csv', parse_dates=['date'])
    cpi = pd.read_csv(INDIR_FRED / 'cpiauscl.csv', parse_dates=['date'])
    cw_period_date = pd.read_csv(INDIR_CW / 'cw_period_date.csv', parse_dates=['date']).set_index('date')
    
    df = add_event_indicators(df)
    df = add_fred(df, mortgage30us, cpi, cw_period_date)
    df = impute_current_upb(df)
    df = compute_rate_spread(df)
    df = compute_rate_gap(df)
    
    for PARAMETER_TYPE, PARAMETERS in PARAMETER_LIST.items():
        df_adl = compute_adl_threshold(df, mortgage30us, parameters=PARAMETERS)
        df_adl = compute_adl_gap(df_adl)
        df_adl = compute_savings(df_adl)
        df_adl = compute_inflation_adjustments(df_adl, cpi, vars=['rate_orig', 'rate_curr', 'upb_orig', 'upb_curr'])
        
        df_adl_full = df_adl[MASK_FULL_SAMPLE]
        df_adl_refi_eligible = df_adl[MASK_REFI_ELIGIBLE]
        
        save_data(
            df_adl_full,
            keys = ['loan_id', 'period'],
            out_file = OUTDIR / f'sflp_sample_processed_{PARAMETER_TYPE}_full.parquet',
            log_file = OUTDIR / f'sflp_sample_processed_{PARAMETER_TYPE}_full.log',
            sortbykey = True
        )

        save_data(
            df_adl_refi_eligible,
            keys = ['loan_id', 'period'],
            out_file = OUTDIR / f'sflp_sample_processed_{PARAMETER_TYPE}_refi_eligible.parquet',
            log_file = OUTDIR / f'sflp_sample_processed_{PARAMETER_TYPE}_refi_eligible.log',
            sortbykey = True
        )

def add_event_indicators(df):
    df['exit_t1'] = np.where(df['time_to_exit'] == 1, 1, 0)
    df['exit_t3'] = np.where(df['time_to_exit'] <= 3, 1, 0)
    df['exit_t6'] = np.where(df['time_to_exit'] <= 6, 1, 0)
    df['exit_t12'] = np.where(df['time_to_exit'] <= 12, 1, 0)
    df['exit_t24'] = np.where(df['time_to_exit'] <= 24, 1, 0)
    return df

def add_fred(df, mortgage30us, cpi, cw_period_date):
    mortgage30us_period = (
        mortgage30us
        .merge(cw_period_date, left_on='date', right_index=True, how='left')
        .drop(columns=['date'])
    )
    df_with_mortgage_rates = (
        df
        .merge(mortgage30us_period.rename(columns={'mortgage_rate': 'rate_mortgage30us'}), on='period', how='left')
        .merge(mortgage30us_period.rename(columns={'mortgage_rate': 'rate_mortgage30us_orig'}), on='period_orig', how='left')
    )
    
    cpi_period = (
        cpi
        .merge(cw_period_date, left_on='date', right_index=True, how='left')
        .drop(columns=['date'])
    )
    df_with_cpi = (
        df_with_mortgage_rates
        .merge(cpi_period, on='period', how='left')
        .assign(inflation_annualized = lambda x: np.log(x["cpi"] / x["cpi"].shift(12)))
    )
    return df_with_cpi

def impute_current_upb(df):
    """Calculate current UPB at a given loan age"""
    df['upb_curr_imputed'] = df.apply(
        lambda row: compute_current_upb(
            loan_age=row['time_from_orig'],
            monthly_interest_rate=((row['rate_orig'] / 100) / 12),
            term=row['term'],
            principal=row['upb_orig']
        ),
        axis=1
    )
    
    mask = (df['upb_curr'] == 0) & (df.groupby('loan_id').cumcount() < 6)
    df.loc[mask, 'upb_curr'] = df.loc[mask, 'upb_curr_imputed']
    df = df.reset_index(drop=True)
    return df

def compute_rate_spread(df):
    df['rate_spread_orig'] = df['rate_orig'] - df['rate_mortgage30us_orig']
    return df

def compute_rate_gap(df, bin_size=0.2, min_bin = -4.0, max_bin = 4.0):
    df['rate_gap'] = df['rate_orig'] - df['rate_mortgage30us']
    df['rate_gap_adj'] = df['rate_orig'] - df['rate_mortgage30us_adj']
    
    bins = np.arange(min_bin, max_bin + bin_size, bin_size)
    bins = np.concatenate([[-np.inf], bins, [np.inf]])
    df['rate_gap_bin'] = pd.cut(df['rate_gap'], bins=bins, right=False, include_lowest=True, labels=False)
    df['rate_gap_adj_bin'] = pd.cut(df['rate_gap_adj'], bins=bins, right=False, include_lowest=True, labels=False)
    return df

def compute_current_upb(loan_age=None, monthly_interest_rate=None, term=None, principal=None):
    return principal * ((1 + monthly_interest_rate)**term - (1 + monthly_interest_rate)**loan_age) / ((1 + monthly_interest_rate)**term - 1)

def compute_adl_threshold(df, mortgage30us, parameters={'ANNUAL_DISCOUNT_RATE': 0.05, 'PROB_MOVE': 0.1, 'MARGINAL_TAX_RATE': 0.28}):
    """Compute the Agarwal, Driscoll, Laibson (2013) optimal refinance thresholds. This is the 'square root rule'."""
    _annual_payment = df.apply(
        lambda row: compute_annuity(
            interest_rate=(row['rate_orig'] / 100),
            term=row['term'] / 12,
            principal=row['upb_orig']
        ),
        axis=1
    )
    _monthly_mortgage_rate_vol = compute_mortgage_rate_vol(mortgage30us)
    _transaction_cost = 0.01 * df["upb_curr"] + 2000
    _lambda = parameters['PROB_MOVE'] + ((_annual_payment / df["upb_curr"]) - (df["rate_orig"] / 100)) + df['inflation_annualized']
    df['adl_threshold'] = 100 * np.sqrt((_monthly_mortgage_rate_vol * _transaction_cost) / (df["upb_curr"] * (1 - parameters['MARGINAL_TAX_RATE']))) * np.sqrt(2 * (parameters['ANNUAL_DISCOUNT_RATE'] + _lambda))
    return df

def compute_annuity(interest_rate=None, term=None, principal=None):
    """Compute monthly mortgage payment"""
    return principal * (interest_rate * (1 + interest_rate)**term) / ((1 + interest_rate)**term - 1)

def compute_mortgage_rate_vol(mortgage30us):
    """Compute monthly mortgage rate volatility"""
    monthly_mortgage_rate_vol = (
        mortgage30us
        .assign(dr=lambda x: x["mortgage_rate"].diff() / 100)
        ["dr"]
        .dropna()
        .std() * np.sqrt(12)
    )
    return monthly_mortgage_rate_vol

def compute_adl_gap(df, bin_size=0.2, min_bin = -4.0, max_bin = 4.0):
    df['adl_gap'] = df['rate_gap'] - df['adl_threshold']
    df['adl_gap_adj'] = df['rate_gap_adj'] - df['adl_threshold']
    
    bins = np.arange(min_bin, max_bin + bin_size, bin_size)
    bins = np.concatenate([[-np.inf], bins, [np.inf]])
    df['adl_gap_bin'] = pd.cut(df['adl_gap'], bins=bins, right=False, include_lowest=True, labels=False)
    df['adl_gap_adj_bin'] = pd.cut(df['adl_gap_adj'], bins=bins, right=False, include_lowest=True, labels=False)
    return df

def compute_savings(group):
    group = group.copy() 
    group['npv_never_refi'] = compute_npv_never_refi(group)
    group['npv_optimal_refi'] = compute_npv_optimal_refi(group)
    group['npv_realized_refi'] = compute_npv_realized_refi(group)
    
    group['savings_optimal_refi'] =  group['npv_never_refi'] - group['npv_optimal_refi']
    group['savings_realized_refi'] = group['npv_never_refi'] - group['npv_realized_refi']
    group['savings_loss'] = group['savings_optimal_refi'] - group['savings_realized_refi']
    return group

def compute_npv_never_refi(group, parameters={'ANNUAL_DISCOUNT_RATE': 0.05}):
    """Compute NPV of no refinance scenario."""
    monthly_discount_rate = parameters['ANNUAL_DISCOUNT_RATE'] / 12
    
    monthly_interest_rate_orig = (group['rate_orig'] / 100) / 12
    term_orig = group['term'].iloc[0]
    upb_orig = group['upb_orig'].iloc[0]
    monthly_payment_orig = compute_annuity(interest_rate=monthly_interest_rate_orig, term=term_orig, principal=upb_orig)
    
    npv = monthly_payment_orig * ((1 - (1 + monthly_discount_rate)**(-term_orig) ) / monthly_discount_rate)
    return npv

def compute_npv_optimal_refi(group, parameters={'ANNUAL_DISCOUNT_RATE': 0.05}):
    """Compute NPV of optimal refinance scenario."""
    monthly_discount_rate = parameters['ANNUAL_DISCOUNT_RATE'] / 12
    
    monthly_interest_rate_orig = (group['rate_orig'] / 100) / 12
    term_orig = group['term'].iloc[0]
    upb_orig = group['upb_orig'].iloc[0]
    monthly_payment_orig = compute_annuity(interest_rate=monthly_interest_rate_orig, term=term_orig, principal=upb_orig)
    
    refi_period = group.loc[group['should_refi'] == 1, 'period'].min()
    monthly_interest_rate_new = ((group.loc[group['period'] == refi_period, 'rate_mortgage30us_adj']) / 100) / 12
    term_new = group.loc[group['period'] == refi_period, 'time_to_maturity'].item()
    upb_new = group.loc[group['period'] == refi_period, 'upb_curr'].item()
    monthly_payment_new = compute_annuity(interest_rate=monthly_interest_rate_new, term=term_new, principal=upb_new)
    

    npv_orig = monthly_payment_orig * ((1 - (1 + monthly_discount_rate)**(-(term_orig - term_new)) ) / monthly_discount_rate)
    npv_new = monthly_payment_new * ((1 - (1 + monthly_discount_rate)**(-term_new) ) / monthly_discount_rate) * ((1 + monthly_discount_rate)**(-(term_orig - term_new)))
    npv_transaction_cost = (0.01 * upb_new + 2000) * ((1 + monthly_discount_rate)**(-(term_orig - term_new)))
    
    npv = npv_orig + npv_new + npv_transaction_cost
    return npv

def compute_npv_realized_refi(group, parameters={'ANNUAL_DISCOUNT_RATE': 0.05}):
    """Compute NPV of the realized refinance scenario."""
    monthly_discount_rate = parameters['ANNUAL_DISCOUNT_RATE'] / 12
    
    monthly_interest_rate_orig = (group['rate_orig'] / 100) / 12
    term_orig = group['term'].iloc[0]
    upb_orig = group['upb_orig'].iloc[0]
    monthly_payment_orig = compute_annuity(interest_rate=monthly_interest_rate_orig, term=term_orig, principal=upb_orig)
    
    refi_period = group['period_exit'].iloc[0] - 1
    monthly_interest_rate_new = ((group.loc[group['period'] == refi_period, 'rate_mortgage30us_adj']) / 100) / 12
    term_new = group.loc[group['period'] == refi_period, 'time_to_maturity'].item()
    upb_new = group.loc[group['period'] == refi_period, 'upb_curr'].item()
    monthly_payment_new = compute_annuity(interest_rate=monthly_interest_rate_new, term=term_new, principal=upb_new)
    
    npv_orig = monthly_payment_orig * ((1 - (1 + monthly_discount_rate)**(-(term_orig - term_new)) ) / monthly_discount_rate)
    npv_new = monthly_payment_new * ((1 - (1 + monthly_discount_rate)**(-term_new) ) / monthly_discount_rate) * ((1 + monthly_discount_rate)**(-(term_orig - term_new)))
    npv_transaction_cost = (0.01 * upb_new + 2000) * ((1 + monthly_discount_rate)**(-(term_orig - term_new)))
    
    npv = npv_orig + npv_new + npv_transaction_cost
    return npv

def compute_inflation_adjustments(df, cpi, vars=None, base_period='2025-01-01'):
    if vars is None or any(var not in df.columns for var in vars):
        raise ValueError("Variable list invalid.")
    
    df['cpi_base'] = cpi.loc[cpi['date'] == base_period, 'cpi'].item()
    for var in vars:
        df[f'{var}_adj'] = df[var] * (df['cpi_base'] / df['cpi'])
    df = df.drop(columns=['cpi_base'])
    return df

if __name__ == '__main__':
    main()

