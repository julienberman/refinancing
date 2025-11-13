import datetime
import numpy as np
import pandas as pd
import janitor
import json
import pyarrow
from pathlib import Path

from source.lib.helpers.process_text import clean_date
from source.lib.helpers.utils import get_quarters
from source.lib.save_data import save_data

def main():
    with open('source/lib/config.json', 'r') as f:
        CONFIG = json.load(f)
    with open('source/lib/schemas.json', 'r') as f:
        SCHEMAS = json.load(f)
    
    INDIR = Path('datastore/raw/fannie_mae/data')
    OUTDIR = Path('datastore/output/derived/fannie_mae/sflp_clean')
    START_DATE, END_DATE = CONFIG['SAMPLE_START'], CONFIG['SAMPLE_END']
    QUARTERS = get_quarters(START_DATE, END_DATE)
    
    
    dfs = []
    for quarter in QUARTERS:
        print(f"Processing {quarter}...")
        
        df = pd.read_csv(
            INDIR / f'{quarter}.csv', sep='|', 
            names=SCHEMAS['fannie_mae'].keys(), 
            dtype=SCHEMAS['fannie_mae'], 
            low_memory=False,
            nrows=10000
        )
        
        df_clean = clean_data(df, quarter=quarter)
        
        save_data(
            df_clean,
            keys=['loan_id', 'date', 'date_acq'],
            out_file=OUTDIR / f'{quarter}.parquet',
        )
        

def clean_data(df, keep_vars=None, quarter=None):
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
        "adj_rem_months": "months_remaining",
        "matr_dt": "date_maturity",
        "oltv": "ltv",
        "csscore_b": "credit_score_orig",
        "csscore_c": "coborrower_credit_score_orig",
        "first_flag": "first_home_buyer",
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
            date = lambda x: clean_date(x['date'], pattern='mmyyyy', aggregation='month'),
            date_orig = lambda x: clean_date(x['date_orig'], pattern='mmyyyy', aggregation='month'),
            date_first_pay = lambda x: clean_date(x['date_first_pay'], pattern='mmyyyy', aggregation='month'),
            date_maturity = lambda x: clean_date(x['date_maturity'], pattern='mmyyyy', aggregation='month'),
            date_zero_balance = lambda x: clean_date(x['date_exit'], pattern='mmyyyy', aggregation='month'),
            date_acq = lambda x: create_acquisition_date(quarter)
        )
    )
    return df_clean

def create_acquisition_date(quarter):
    acquisition_year = quarter[:4]
    acquisition_quarter = quarter[4:]
    month_map = {
        'Q1': '03',
        'Q2': '06',
        'Q3': '09',
        'Q4': '12'
    }
    acquisition_date = clean_date(f'{month_map[acquisition_quarter]}{acquisition_year}', pattern='mmyyyy', aggregation='month')
    print(acquisition_date)
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
    mask = (df['exit_code'] == 'prepaid') & (df['date_exit'] == df['date_maturity'])
    df.loc[mask, 'exit_code'] = 'matured'
    return df

if __name__ == '__main__':
    main()

