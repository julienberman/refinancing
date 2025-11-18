import datetime
import numpy as np
import pandas as pd
import janitor
import json
import pyarrow
from pathlib import Path
import glob

from source.lib.helpers.process_text import clean_date, clean_text
from source.lib.helpers.utils import get_quarters
from source.lib.save_data import save_data

def main():
    with open('source/lib/config.json', 'r') as f:
        CONFIG = json.load(f)
    with open('source/lib/schemas.json', 'r') as f:
        SCHEMAS = json.load(f)
    
    INDIR = Path('datastore/raw/fannie_mae/data')
    INDIR_CW = Path('datastore/raw/crosswalks/data')
    INDIR_MORTGAGE_RATES = Path('output/derived/mortgage_rates')
    OUTDIR = Path('datastore/output/derived/fannie_mae/sflp_clean')
    LOGDIR = Path('output/derived/fannie_mae/sflp_clean')
    START_DATE, END_DATE = CONFIG['SAMPLE_START'], CONFIG['SAMPLE_END']
    QUARTERS = get_quarters(START_DATE, END_DATE)
    METADATA = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in SCHEMAS['fannie_mae'].items()})
    
    cw_state_county = pd.read_csv(INDIR_CW / 'cw_state_county.csv')
    cw_period_date = pd.read_csv(INDIR_CW / 'cw_period_date.csv', parse_dates=['date']).set_index('date')
    mortgage30us = pd.read_csv(INDIR_MORTGAGE_RATES / 'mortgage30us.csv', parse_dates=['date'])
    
    for quarter in QUARTERS:
        print(f"Processing {quarter}...")
        n_chunks = len(list(glob.glob(str(INDIR / f'{quarter}/*.parquet'))))
        df = pd.read_parquet(INDIR / f'{quarter}')

        df_clean = clean_data(df, cw_period_date, quarter=quarter)
        df_with_fips = add_fips(df_clean, cw_state_county)
        df_with_mortgage_rates = add_mortgage_rate(df_with_fips, mortgage30us, cw_period_date)
        df_with_indicators = add_event_indicators(df_with_mortgage_rates)
        df_with_bins = bin_rate_gap(df_with_indicators)
        df_finalized = finalize_data(df_with_bins)
        
        save_data(
            df_finalized,
            keys=['loan_id', 'period'],
            out_file=OUTDIR / f'{quarter}',
            log_file=LOGDIR / f'{quarter}.log',
            sortbykey=True,
            n_partitions=n_chunks
        )
        
def clean_data(df, cw_period_date, keep_vars=None, quarter=None):
    keep_vars = keep_vars or [
        "LOAN_ID", "ACT_PERIOD", "ORIG_RATE", "CURR_RATE", "ORIG_UPB", "CURRENT_UPB",
        "ORIG_TERM", "ORIG_DATE", "FIRST_PAY", "LOAN_AGE", "ADJ_REM_MONTHS", "MATR_DT",
        "OLTV", "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG", "PURPOSE",
        "STATE", "MSA", "ZIP", "PRODUCT", "DLQ_STATUS", "ZERO_BAL_CODE", "ZB_DTE",
        "LAST_UPB", "CURR_SCOREB", "CURR_SCOREC"
    ]
    
    rename_table = {
        "act_period": "date",
        "orig_rate": "rate_orig",
        "curr_rate": "rate_curr",
        "orig_upb": "upb_orig",
        "current_upb": "upb_curr",
        "orig_term": "term",
        "orig_date": "date_orig",
        "first_pay": "date_first_pay",
        "loan_age": "time_from_orig",
        "adj_rem_months": "time_to_maturity",
        "matr_dt": "date_maturity",
        "oltv": "ltv",
        "num_bo": "n_borrowers",
        "cscore_b": "credit_score_orig",
        "cscore_c": "coborrower_credit_score_orig",
        "first_flag": "first_home_buyer",
        "state": "state_abbr",
        "product": "mortgage_type",
        "zero_bal_code": "exit_code",
        "zb_dte": "date_exit",
        "last_upb": "upb_last",
        "curr_scoreb": "credit_score_curr",
        "curr_scorec": "coborrower_credit_score_curr"
    }
    
    df_clean = (
        df
        .select(columns=keep_vars)
        .clean_names()
        .rename(columns=rename_table)
        .assign(
            date_orig = lambda x: clean_date(x['date_orig'], pattern='mmyyyy'),
            date_first_pay = lambda x: clean_date(x['date_first_pay'], pattern='mmyyyy'),
            date_maturity = lambda x: clean_date(x['date_maturity'], pattern='mmyyyy'),
            date_exit = lambda x: clean_date(x['date_exit'], pattern='mmyyyy'),
            date_acq = lambda x: create_acquisition_date(quarter),
            date = lambda x: clean_date(x['date'], pattern='mmyyyy')
        )
        .merge(cw_period_date, left_on='date_orig', right_index=True, how='left').rename(columns={'period': 'period_orig'})
        .merge(cw_period_date, left_on='date_first_pay', right_index=True, how='left').rename(columns={'period': 'period_first_pay'})
        .merge(cw_period_date, left_on='date_maturity', right_index=True, how='left').rename(columns={'period': 'period_maturity'})
        .merge(cw_period_date, left_on='date_exit', right_index=True, how='left').rename(columns={'period': 'period_exit'})
        .merge(cw_period_date, left_on='date_acq', right_index=True, how='left').rename(columns={'period': 'period_acq'})
        .merge(cw_period_date, left_on='date', right_index=True, how='left')
        .drop(columns=['date_orig', 'date_first_pay', 'date_maturity', 'date_exit', 'date_acq', 'date'])
        .assign(
            first_home_buyer = lambda x: x['first_home_buyer'].map({'Y': 1, 'N': 0}),
            mortgage_type = lambda x: x['mortgage_type'].map({'FRM': 'fixed', 'ARM': 'adjustable'})
        )
    )
    
    df_with_exit_codes = clean_exit_code(df_clean)
    
    return df_with_exit_codes

def create_acquisition_date(quarter):
    acquisition_year = quarter[:4]
    acquisition_quarter = quarter[4:]
    month_map = {
        'Q1': '03',
        'Q2': '06',
        'Q3': '09',
        'Q4': '12'
    }
    acquisition_date = clean_date(f'{month_map[acquisition_quarter]}{acquisition_year}', pattern='mmyyyy')
    return acquisition_date
    
def clean_exit_code(df):
    recode_map = {
        '01': 'prepaid',
        '02': 'third_party_sale',
        '03': 'short_sale',
        '06': 'repurchased',
        '09': 'deed_in_lieu',
        '15': 'non_performing_note_sale',
        '16': 'reperforming_note_sale',
        '96': 'removal',
        '97': 'delinquency',
        '98': 'other'
    }
    
    df['exit_code'] = df['exit_code'].map(recode_map)
    mask = (df['exit_code'] == 'prepaid') & (df['period_exit'] == df['period_maturity'])
    df.loc[mask, 'exit_code'] = 'matured'
    return df

def add_fips(df, cw_state_county):
    cw_state = (
        cw_state_county
        .drop_duplicates(subset=["fips_state"])
        .select(columns=['state', 'state_abbr', 'fips_state'])
    )
    df_with_fips = (
        df
        .rename(columns={"state": "state_abbr"})
        .assign(state_abbr = lambda x: clean_text(x['state_abbr'], lower=True))
        .merge(cw_state, how='left', on='state_abbr')
    )
    return df_with_fips

def add_mortgage_rate(df, mortgage30us, cw_period_date):
    mortgage30us_period = (
        mortgage30us
        .merge(cw_period_date, left_on='date', right_index=True, how='left')
        .drop(columns=['date'])
        .rename(columns={'mortgage_rate': 'rate_mortgage30us'})
    )
    
    df_with_mortgage_rates = (
        df
        .merge(mortgage30us_period, on='period', how='left')
        .assign(rate_gap = lambda x: x['rate_curr'] - x['rate_mortgage30us'])
    )
    return df_with_mortgage_rates

def add_event_indicators(df):
    df_with_indicators = (
        df
        .assign(period_exit = lambda x: x.groupby('loan_id')['period_exit'].ffill().bfill())
        .assign(exit_code = lambda x: x.groupby('loan_id')['exit_code'].ffill().bfill())
        .assign(time_to_exit = lambda x: x['period_exit'] - x['period'])
        .assign(exit_t1 = lambda x: np.where(x['time_to_exit'] == 1, 1, 0))
        .assign(exit_t3 = lambda x: np.where(x['time_to_exit'] <= 3, 1, 0))
        .assign(exit_t6 = lambda x: np.where(x['time_to_exit'] <= 6, 1, 0))
        .assign(exit_t12 = lambda x: np.where(x['time_to_exit'] <= 12, 1, 0))
        .assign(exit_t24 = lambda x: np.where(x['time_to_exit'] <= 24, 1, 0))
    )

    return df_with_indicators

def bin_rate_gap(df, width=0.2):
    df_with_bins = df.copy()
    bins = np.arange(-4.0, 4.0 + width, width)
    bins = np.concatenate([[-np.inf], bins, [np.inf]])
    df_with_bins['rate_gap_bin'] = pd.cut(df['rate_gap'], bins=bins, right=False, include_lowest=True, labels=False)
    return df_with_bins

def finalize_data(df):
    columns = [
        "loan_id", "period", "rate_orig", "rate_curr", "rate_mortgage30us", "rate_gap", "rate_gap_bin", "upb_orig", "upb_curr", "ltv", "dti", 
        "n_borrowers", "term", "period_orig", "period_acq", "period_first_pay", "time_from_orig", "time_to_maturity", 
        "period_maturity", "time_to_exit", "period_exit", "exit_code", "exit_t1", "exit_t3", "exit_t6", "exit_t12", 
        "exit_t24", "upb_last", "credit_score_curr", "coborrower_credit_score_curr", "credit_score_orig", 
        "coborrower_credit_score_orig", "first_home_buyer", "mortgage_type", "purpose", "dlq_status", 
        "state", "state_abbr", "fips_state", "msa", "zip"
    ]
    
    df = df.select(columns=columns)
    return df

if __name__ == '__main__':
    main()

