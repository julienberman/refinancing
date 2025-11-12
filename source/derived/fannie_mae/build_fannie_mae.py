import numpy as np
import pandas as pd
import janitor
import json
from pathlib import Path

from source.lib.helpers.utils import get_quarters
from source.lib.save_data import save_data

def main():
    with open('source/lib/config.json', 'r') as f:
        CONFIG = json.load(f)
    with open('source/lib/schemas.json', 'r') as f:
        SCHEMAS = json.load(f)
    
    INDIR = Path('datastore/raw/fannie_mae/data')
    OUTDIR = Path('output/derived/fannie_mae')
    START_DATE, END_DATE = CONFIG['SAMPLE_START'], CONFIG['SAMPLE_END']
    QUARTERS = get_quarters(START_DATE, END_DATE)
    
    
    dfs = []
    for quarter in QUARTERS:
        print(f"Processing {quarter}...")
        
        df = load_file(in_file = INDIR / quarter / '.csv', schema = SCHEMAS['fannie_mae'])
        df_base = prepare_base_data(df)
        acquisition_file = create_acquisition_data(df_base)
        performance_file = create_performance_data(df_base)
        base_table_1 = create_base_table_1(acquisition_file, quarter)
        base_table_2 = create_base_table_2(base_table_1, performance_file)
        base_table_3 = create_base_table_3(base_table_2, performance_file)
        base_table_4 = create_base_table_4(base_table_3, performance_file)
        base_table_5 = create_base_table_5(base_table_4)
        base_table_6 = create_base_table_6(base_table_5, base_table_1, performance_file)
        output = create_final_output(base_table_6)
        
        dfs.append(output)
    
    df_combined = pd.concat(dfs)
    
    save_data(
        df_combined,
        keys = ...,
        out_file = OUTDIR / 'fannie_mae.parquet',
        log_file = OUTDIR / 'fannie_mae.log',
        sortbykey = False
    )

def load_file(in_file=None, schema=None):
    if not in_file or not schema:
        raise ValueError('Please provide the following parameters: `in_file`, `schema`')
    
    df = pd.read_csv(in_file, sep='|', names=schema.keys(), dtype=schema, low_memory=False)
    df['ORIG_RATE'] = pd.to_numeric(df['ORIG_RATE'], errors='coerce')
    df['CURR_RATE'] = pd.to_numeric(df['CURR_RATE'], errors='coerce')
    
    return df

def prepare_base_data(df):
    keep_cols = [
        'LOAN_ID', 'ACT_PERIOD', 'CHANNEL', 'SELLER', 'SERVICER',
        'ORIG_RATE', 'CURR_RATE', 'ORIG_UPB', 'CURRENT_UPB', 'ORIG_TERM',
        'ORIG_DATE', 'FIRST_PAY', 'LOAN_AGE', 'REM_MONTHS', 'ADJ_REM_MONTHS',
        'MATR_DT', 'OLTV', 'OCLTV', 'NUM_BO', 'DTI',
        'CSCORE_B', 'CSCORE_C', 'FIRST_FLAG', 'PURPOSE', 'PROP',
        'NO_UNITS', 'OCC_STAT', 'STATE', 'MSA', 'ZIP',
        'MI_PCT', 'PRODUCT', 'DLQ_STATUS', 'MOD_FLAG', 'Zero_Bal_Code',
        'ZB_DTE', 'LAST_PAID_INSTALLMENT_DATE', 'FORECLOSURE_DATE', 'DISPOSITION_DATE',
        'FORECLOSURE_COSTS', 'PROPERTY_PRESERVATION_AND_REPAIR_COSTS', 'ASSET_RECOVERY_COSTS',
        'MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS', 'ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY',
        'NET_SALES_PROCEEDS', 'CREDIT_ENHANCEMENT_PROCEEDS', 'REPURCHASES_MAKE_WHOLE_PROCEEDS',
        'OTHER_FORECLOSURE_PROCEEDS', 'NON_INTEREST_BEARING_UPB', 'PRINCIPAL_FORGIVENESS_AMOUNT',
        'RELOCATION_MORTGAGE_INDICATOR', 'MI_TYPE', 'SERV_IND', 'RPRCH_DTE', 'LAST_UPB'
    ]
    
    df_clean = (
        df
        .select(columns=keep_cols)
        .assign(
            repch_flag=lambda x: np.where(x['RPRCH_DTE'].notna(), 1, 0),
            ACT_PERIOD=lambda x: x['ACT_PERIOD'].str[2:6] + '-' + x['ACT_PERIOD'].str[0:2] + '-01',
            FIRST_PAY=lambda x: x['FIRST_PAY'].str[2:6] + '-' + x['FIRST_PAY'].str[0:2] + '-01',
            ORIG_DATE=lambda x: x['ORIG_DATE'].str[2:6] + '-' + x['ORIG_DATE'].str[0:2] + '-01'
        )
        .sort_values(['LOAN_ID', 'ACT_PERIOD'])
    )
    return df_clean

def create_acquisition_data(df_base):
    keep_cols = [
        'LOAN_ID', 'ACT_PERIOD', 'CHANNEL', 'SELLER', 'ORIG_RATE', 'ORIG_UPB',
        'ORIG_TERM', 'ORIG_DATE', 'FIRST_PAY', 'OLTV',
        'OCLTV', 'NUM_BO', 'DTI', 'CSCORE_B', 'CSCORE_C',
        'FIRST_FLAG', 'PURPOSE', 'PROP', 'NO_UNITS', 'OCC_STAT',
        'STATE', 'ZIP', 'MI_PCT', 'PRODUCT', 'MI_TYPE',
        'RELOCATION_MORTGAGE_INDICATOR'
    ]
    
    rename_table = {
        'CHANNEL': 'ORIG_CHN',
        'ORIG_RATE': 'orig_rt',
        'ORIG_UPB': 'orig_amt',
        'ORIG_TERM': 'orig_trm',
        'ORIG_DATE': 'orig_date',
        'FIRST_PAY': 'first_pay',
        'OLTV': 'oltv',
        'OCLTV': 'ocltv',
        'NUM_BO': 'num_bo',
        'DTI': 'dti',
        'FIRST_FLAG': 'FTHB_FLG',
        'PURPOSE': 'purpose',
        'PROP': 'PROP_TYP',
        'NO_UNITS': 'NUM_UNIT',
        'OCC_STAT': 'occ_stat',
        'STATE': 'state',
        'ZIP': 'zip_3',
        'MI_PCT': 'mi_pct',
        'PRODUCT': 'prod_type',
        'RELOCATION_MORTGAGE_INDICATOR': 'relo_flg'
    }

    acquisition_file = (
        df_base
        .select(columns=keep_cols)
        .rename(columns=rename_table)
    )
    
    # Get first period for each loan
    acquisition_file = (
        acquisition_file
        .groupby('LOAN_ID')
        .agg(first_period=('ACT_PERIOD', 'max'))
        .reset_index()
        .merge(acquisition_file, left_on=['LOAN_ID', 'first_period'], right_on=['LOAN_ID', 'ACT_PERIOD'], how='left')
        .select(columns=[
            'LOAN_ID', 'ORIG_CHN', 'SELLER', 'orig_rt', 'orig_amt',
            'orig_trm', 'orig_date', 'first_pay', 'oltv',
            'ocltv', 'num_bo', 'dti', 'CSCORE_B', 'CSCORE_C',
            'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat',
            'state', 'zip_3', 'mi_pct', 'prod_type', 'MI_TYPE',
            'relo_flg'
        ])
    )
    
    return acquisition_file


def create_performance_data(df_base):
    keep_cols = [
        'LOAN_ID', 'ACT_PERIOD', 'SERVICER', 'CURR_RATE', 'CURRENT_UPB',
        'LOAN_AGE', 'REM_MONTHS', 'ADJ_REM_MONTHS', 'MATR_DT', 'MSA',
        'DLQ_STATUS', 'MOD_FLAG', 'Zero_Bal_Code', 'ZB_DTE', 'LAST_PAID_INSTALLMENT_DATE',
        'FORECLOSURE_DATE', 'DISPOSITION_DATE', 'FORECLOSURE_COSTS', 
        'PROPERTY_PRESERVATION_AND_REPAIR_COSTS', 'ASSET_RECOVERY_COSTS',
        'MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS', 'ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY', 
        'NET_SALES_PROCEEDS', 'CREDIT_ENHANCEMENT_PROCEEDS', 'REPURCHASES_MAKE_WHOLE_PROCEEDS',
        'OTHER_FORECLOSURE_PROCEEDS', 'NON_INTEREST_BEARING_UPB', 
        'PRINCIPAL_FORGIVENESS_AMOUNT', 'repch_flag', 'LAST_UPB'
    ]
    
    rename_table = {
        'ACT_PERIOD': 'period',
        'SERVICER': 'servicer',
        'CURR_RATE': 'curr_rte',
        'CURRENT_UPB': 'act_upb',
        'LOAN_AGE': 'loan_age',
        'REM_MONTHS': 'rem_mths',
        'ADJ_REM_MONTHS': 'adj_rem_months',
        'MATR_DT': 'maturity_date',
        'MSA': 'msa',
        'DLQ_STATUS': 'dlq_status',
        'MOD_FLAG': 'mod_ind',
        'Zero_Bal_Code': 'z_zb_code',
        'ZB_DTE': 'zb_date',
        'LAST_PAID_INSTALLMENT_DATE': 'lpi_dte',
        'FORECLOSURE_DATE': 'fcc_dte',
        'DISPOSITION_DATE': 'disp_dte',
        'FORECLOSURE_COSTS': 'FCC_COST',
        'PROPERTY_PRESERVATION_AND_REPAIR_COSTS': 'PP_COST',
        'ASSET_RECOVERY_COSTS': 'AR_COST',
        'MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS': 'IE_COST',
        'ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY': 'TAX_COST',
        'NET_SALES_PROCEEDS': 'NS_PROCS',
        'CREDIT_ENHANCEMENT_PROCEEDS': 'CE_PROCS',
        'REPURCHASES_MAKE_WHOLE_PROCEEDS': 'RMW_PROCS',
        'OTHER_FORECLOSURE_PROCEEDS': 'O_PROCS',
        'NON_INTEREST_BEARING_UPB': 'non_int_upb',
        'PRINCIPAL_FORGIVENESS_AMOUNT': 'prin_forg_upb',
        'LAST_UPB': 'zb_upb'
    }

    performance_file = (
        df_base
        .select(columns=keep_cols)
        .rename(columns=rename_table)
        .assign(
            maturity_date=lambda x: np.where(
                x['maturity_date'] != '',
                x['maturity_date'].str[2:6] + '-' + x['maturity_date'].str[0:2] + '-01',
                x['maturity_date']
            ),
            zb_date=lambda x: np.where(
                x['zb_date'] != '',
                x['zb_date'].str[2:6] + '-' + x['zb_date'].str[0:2] + '-01',
                x['zb_date']
            ),
            lpi_dte=lambda x: np.where(
                x['lpi_dte'] != '',
                x['lpi_dte'].str[2:6] + '-' + x['lpi_dte'].str[0:2] + '-01',
                x['lpi_dte']
            ),
            fcc_dte=lambda x: np.where(
                x['fcc_dte'] != '',
                x['fcc_dte'].str[2:6] + '-' + x['fcc_dte'].str[0:2] + '-01',
                x['fcc_dte']
            ),
            disp_dte=lambda x: np.where(
                x['disp_dte'] != '',
                x['disp_dte'].str[2:6] + '-' + x['disp_dte'].str[0:2] + '-01',
                x['disp_dte']
            )
        )
    )
    
    return performance_file


def create_base_table_1(acquisition_file, quarter):
    """
    Create first base table with acquisition fields plus AQSN_DTE and recodes.
    """
    # Parse acquisition date from quarter
    acquisition_year = quarter[:4]
    acquisition_qtr = quarter[4:6]
    if acquisition_qtr == 'Q1':
        acquisition_month = '03'
    elif acquisition_qtr == 'Q2':
        acquisition_month = '06'
    elif acquisition_qtr == 'Q3':
        acquisition_month = '09'
    else:
        acquisition_month = '12'
    acquisition_date = f"{acquisition_year}-{acquisition_month}-01"
    
    # Rename date fields
    acquisition_file = acquisition_file.rename(columns={
        'orig_date': 'ORIG_DTE',
        'first_pay': 'FRST_DTE'
    })
    
    base_table_1 = acquisition_file.assign(
        AQSN_DTE=acquisition_date,
        MI_TYPE=lambda x: np.select(
            [x['MI_TYPE'] == '1', x['MI_TYPE'] == '2', x['MI_TYPE'] == '3'],
            ['BPMI', 'LPMI', 'IPMI'],
            default='None'
        ),
        ocltv=lambda x: np.where(x['ocltv'].isna(), x['oltv'], x['ocltv'])
    )
    
    return base_table_1


def create_base_table_2(base_table_1, performance_file):
    """
    Create second base table with latest-available or aggregated performance data.
    
    """
    # Last activity date
    last_act_dte_table = (
        performance_file
        .groupby('LOAN_ID')
        .agg(LAST_ACTIVITY_DATE=('period', 'max'))
        .reset_index()
    )
    
    # Last UPB
    last_upb_table = (
        performance_file
        .groupby('LOAN_ID')
        .apply(lambda x: x.loc[x['period'] == x['period'].max()])
        .reset_index(drop=True)
        .assign(LAST_UPB=lambda x: np.where(x['zb_upb'].notna(), x['zb_upb'], x['act_upb']))
        .select(columns=['LOAN_ID', 'LAST_UPB'])
    )
    
    # Last rate
    last_rt_table = (
        performance_file
        .query('curr_rte.notna()')
        .groupby('LOAN_ID')
        .agg(LAST_RT_DATE=('period', 'max'))
        .reset_index()
        .merge(performance_file, left_on=['LOAN_ID', 'LAST_RT_DATE'], right_on=['LOAN_ID', 'period'], how='left')
        .assign(LAST_RT=lambda x: x['curr_rte'].round(3))
        .select(columns=['LOAN_ID', 'LAST_RT'])
    )
    
    # Zero balance code
    zb_code_table = (
        performance_file
        .query('z_zb_code != ""')
        .groupby('LOAN_ID')
        .agg(zb_code_dt=('period', 'max'))
        .reset_index()
        .merge(performance_file, left_on=['LOAN_ID', 'zb_code_dt'], right_on=['LOAN_ID', 'period'], how='left')
        .rename(columns={'z_zb_code': 'zb_code'})
        .select(columns=['LOAN_ID', 'zb_code'])
    )
    
    # Combine max tables
    max_table = (
        last_act_dte_table
        .merge(performance_file, on=['LOAN_ID', 'LAST_ACTIVITY_DATE'], how='left')
        .merge(last_upb_table, on='LOAN_ID', how='left')
        .merge(last_rt_table, on='LOAN_ID', how='left')
        .merge(zb_code_table, on='LOAN_ID', how='left')
    )
    
    # Servicer
    servicer_table = (
        performance_file
        .query('servicer != ""')
        .groupby('LOAN_ID')
        .agg(servicer_period=('period', 'max'))
        .reset_index()
        .merge(performance_file, left_on=['LOAN_ID', 'servicer_period'], 
               right_on=['LOAN_ID', 'period'], how='left')
        .assign(SERVICER=lambda x: x['servicer'])
        .select(columns=['LOAN_ID', 'SERVICER'])
    )
    
    # Non-interest bearing UPB
    non_int_upb_table = (
        performance_file
        .groupby('LOAN_ID')
        .apply(lambda x: x.iloc[-2] if len(x) > 1 else x.iloc[-1])
        .reset_index(drop=True)
        .assign(NON_INT_UPB=lambda x: np.where(x['non_int_upb'].isna(), 0, x['non_int_upb']))
        .select(columns=['LOAN_ID', 'NON_INT_UPB'])
    )
    
    base_table_2 = (
        base_table_1
        .merge(max_table, on='LOAN_ID', how='left')
        .merge(servicer_table, on='LOAN_ID', how='left')
        .merge(non_int_upb_table, on='LOAN_ID', how='left')
    )
    
    return base_table_2


def create_base_table_3(base_table_2, performance_file):
    """
    Create third base table with foreclosure/disposition data.
    """
    fcc_table = (
        performance_file
        .query('lpi_dte.notna() & fcc_dte.notna() & disp_dte.notna()')
        .groupby('LOAN_ID')
        .agg(
            LPI_DTE=('lpi_dte', 'max'),
            FCC_DTE=('fcc_dte', 'max'),
            DISP_DTE=('disp_dte', 'max')
        )
        .reset_index()
    )
    
    base_table_3 = base_table_2.merge(fcc_table, on='LOAN_ID', how='left')
    
    return base_table_3


def create_base_table_4(base_table_3, performance_file):
    """
    Create fourth base table with first delinquency occurrence and modification data.
    """
    # Create slim performance file for DQ calculations
    keep_cols = [
        'LOAN_ID', 'period', 'dlq_status', 'z_zb_code',
        'act_upb', 'zb_upb', 'mod_ind', 'maturity_date', 'rem_mths'
    ]
    slim_performance_file = (
        performance_file
        .select(columns=keep_cols)
        .assign(dlq_status=lambda x: np.where(x['dlq_status'] == 'XX', '999', x['dlq_status']))
        .assign(dlq_status=lambda x: pd.to_numeric(x['dlq_status'], errors='coerce'))
    )
    
    # First 30-day delinquency
    f30_table = (
        slim_performance_file
        .query('dlq_status >= 1 & dlq_status < 999 & z_zb_code == ""')
        .groupby('LOAN_ID')
        .agg(F30_DTE=('period', 'min'))
        .reset_index()
        .merge(slim_performance_file, left_on=['LOAN_ID', 'F30_DTE'], right_on=['LOAN_ID', 'period'], how='left')
        .rename(columns={'act_upb': 'F30_UPB'})
        .select(columns=['LOAN_ID', 'F30_DTE', 'F30_UPB'])
    )
    
    # First 60-day delinquency
    f60_table = (
        slim_performance_file
        .query('dlq_status >= 2 & dlq_status < 999 & z_zb_code == ""')
        .groupby('LOAN_ID')
        .agg(F60_DTE=('period', 'min'))
        .reset_index()
        .merge(slim_performance_file, left_on=['LOAN_ID', 'F60_DTE'], right_on=['LOAN_ID', 'period'], how='left')
        .rename(columns={'act_upb': 'F60_UPB'})
        .select(columns=['LOAN_ID', 'F60_DTE', 'F60_UPB'])
    )
    
    # First 90-day delinquency
    f90_table = (
        slim_performance_file
        .query('dlq_status >= 3 & dlq_status < 999 & z_zb_code == ""')
        .groupby('LOAN_ID')
        .agg(F90_DTE=('period', 'min'))
        .reset_index()
        .merge(slim_performance_file, left_on=['LOAN_ID', 'F90_DTE'], right_on=['LOAN_ID', 'period'], how='left')
        .rename(columns={'act_upb': 'F90_UPB'})
        .select(columns=['LOAN_ID', 'F90_DTE', 'F90_UPB'])
    )
    
    # First 120-day delinquency
    f120_table = (
        slim_performance_file
        .query('dlq_status >= 4 & dlq_status < 999 & z_zb_code == ""')
        .groupby('LOAN_ID')
        .agg(F120_DTE=('period', 'min'))
        .reset_index()
        .merge(slim_performance_file, left_on=['LOAN_ID', 'F120_DTE'], right_on=['LOAN_ID', 'period'], how='left')
        .rename(columns={'act_upb': 'F120_UPB'})
        .select(columns=['LOAN_ID', 'F120_DTE', 'F120_UPB'])
    )
    
    # First 180-day delinquency
    f180_table = (
        slim_performance_file
        .query('dlq_status >= 6 & dlq_status < 999 & z_zb_code == ""')
        .groupby('LOAN_ID')
        .agg(F180_DTE=('period', 'min'))
        .reset_index()
        .merge(slim_performance_file, left_on=['LOAN_ID', 'F180_DTE'], right_on=['LOAN_ID', 'period'], how='left')
        .rename(columns={'act_upb': 'F180_UPB'})
        .select(columns=['LOAN_ID', 'F180_DTE', 'F180_UPB'])
    )
    
    # First credit event
    fce_table = (
        slim_performance_file
        .query('(z_zb_code == "02" | z_zb_code == "03" | z_zb_code == "09" | z_zb_code == "15") | (dlq_status >= 6 & dlq_status < 999)')
        .groupby('LOAN_ID')
        .agg(FCE_DTE=('period', 'min'))
        .reset_index()
        .merge(slim_performance_file, left_on=['LOAN_ID', 'FCE_DTE'], right_on=['LOAN_ID', 'period'], how='left')
        .assign(FCE_UPB=lambda x: x['zb_upb'] + x['act_upb'])
        .select(columns=['LOAN_ID', 'FCE_DTE', 'FCE_UPB'])
    )
    
    # First modification
    fmod_dte_table = (
        slim_performance_file
        .query('mod_ind == "Y" & z_zb_code == ""')
        .groupby('LOAN_ID')
        .agg(FMOD_DTE=('period', 'min'))
        .reset_index()
    )
    
    fmod_table = (
        slim_performance_file
        .query('mod_ind == "Y" & z_zb_code == ""')
        .merge(fmod_dte_table, on='LOAN_ID', how='left')
        .assign(
            period_months=lambda x: (x['period'].str[:4].astype(int) * 12 + x['period'].str[5:7].astype(int)),
            fmod_months=lambda x: (x['FMOD_DTE'].str[:4].astype(int) * 12 + x['FMOD_DTE'].str[5:7].astype(int))
        )
        .query('period_months <= fmod_months + 3')
        .groupby('LOAN_ID')
        .agg(FMOD_UPB=('act_upb', 'max'))
        .reset_index()
        .merge(fmod_dte_table, on='LOAN_ID', how='left')
        .merge(slim_performance_file, left_on=['LOAN_ID', 'FMOD_DTE'], right_on=['LOAN_ID', 'period'], how='left')
        .select(columns=['LOAN_ID', 'FMOD_DTE', 'FMOD_UPB', 'maturity_date'])
    )
    
    # Number of periods to 120-day delinquency
    num_120_table = (
        f120_table
        .merge(base_table_3.select(columns=['LOAN_ID', 'FRST_DTE']), on='LOAN_ID', how='left')
        .assign(
            z_num_periods_120=lambda x: (
                (x['F120_DTE'].str[:4].astype(int) * 12 + x['F120_DTE'].str[5:7].astype(int)) -
                (x['FRST_DTE'].str[:4].astype(int) * 12 + x['FRST_DTE'].str[5:7].astype(int)) + 1
            )
        )
        .select(columns=['LOAN_ID', 'z_num_periods_120'])
    )
    
    # Modification term change
    orig_maturity_table = (
        slim_performance_file
        .query('maturity_date.notna()')
        .groupby('LOAN_ID')
        .agg(maturity_date_period=('period', 'min'))
        .reset_index()
        .merge(slim_performance_file, left_on=['LOAN_ID', 'maturity_date_period'], 
               right_on=['LOAN_ID', 'period'], how='left')
        .rename(columns={'maturity_date': 'orig_maturity_date'})
        .select(columns=['LOAN_ID', 'orig_maturity_date'])
    )
    
    trm_chng_table = (
        slim_performance_file
        .sort_values(['LOAN_ID', 'period'])
        .groupby('LOAN_ID')
        .assign(
            prev_rem_mths=lambda x: x['rem_mths'].shift(1),
            trm_chng=lambda x: x['rem_mths'] - x['prev_rem_mths'],
            did_trm_chng=lambda x: np.where(x['trm_chng'] >= 0, 1, 0)
        )
        .query('did_trm_chng == 1')
        .groupby('LOAN_ID')
        .agg(trm_chng_dt=('period', 'min'))
        .reset_index()
    )
    
    modtrm_table = (
        fmod_table
        .merge(orig_maturity_table, on='LOAN_ID', how='left')
        .merge(trm_chng_table, on='LOAN_ID', how='left')
        .assign(MODTRM_CHNG=lambda x: np.where(
            (x['maturity_date'] != x['orig_maturity_date']) | x['trm_chng_dt'].notna(), 
            1, 0
        ))
        .select(columns=['LOAN_ID', 'MODTRM_CHNG'])
    )
    
    # Modification UPB change
    pre_mod_upb_table = (
        slim_performance_file
        .merge(fmod_table, on='LOAN_ID', how='left')
        .query('period < FMOD_DTE')
        .groupby('LOAN_ID')
        .agg(pre_mod_period=('period', 'max'))
        .reset_index()
        .merge(slim_performance_file, left_on=['LOAN_ID', 'pre_mod_period'], 
               right_on=['LOAN_ID', 'period'], how='left')
        .rename(columns={'act_upb': 'pre_mod_upb'})
    )
    
    modupb_table = (
        fmod_table
        .merge(pre_mod_upb_table, on='LOAN_ID', how='left')
        .assign(MODUPB_CHNG=lambda x: np.where(x['FMOD_UPB'] >= x['pre_mod_upb'], 1, 0))
        .select(columns=['LOAN_ID', 'MODUPB_CHNG'])
    )
    
    # Merge all tables
    base_table_4 = (
        base_table_3
        .merge(f30_table, on='LOAN_ID', how='left')
        .merge(f60_table, on='LOAN_ID', how='left')
        .merge(f90_table, on='LOAN_ID', how='left')
        .merge(f120_table, on='LOAN_ID', how='left')
        .merge(f180_table, on='LOAN_ID', how='left')
        .merge(fce_table, on='LOAN_ID', how='left')
        .merge(fmod_table, on='LOAN_ID', how='left')
        .merge(num_120_table, on='LOAN_ID', how='left')
        .merge(modtrm_table, on='LOAN_ID', how='left')
        .merge(modupb_table, on='LOAN_ID', how='left')
        .assign(
            F30_UPB=lambda x: np.where((x['F30_UPB'].isna()) & (x['F30_DTE'].notna()), x['orig_amt'], x['F30_UPB']),
            F60_UPB=lambda x: np.where((x['F60_UPB'].isna()) & (x['F60_DTE'].notna()), x['orig_amt'], x['F60_UPB']),
            F90_UPB=lambda x: np.where((x['F90_UPB'].isna()) & (x['F90_DTE'].notna()), x['orig_amt'], x['F90_UPB']),
            F120_UPB=lambda x: np.where((x['F120_UPB'].isna()) & (x['F120_DTE'].notna()), x['orig_amt'], x['F120_UPB']),
            F180_UPB=lambda x: np.where((x['F180_UPB'].isna()) & (x['F180_DTE'].notna()), x['orig_amt'], x['F180_UPB']),
            FCE_UPB=lambda x: np.where((x['FCE_UPB'].isna()) & (x['FCE_DTE'].notna()), x['orig_amt'], x['FCE_UPB'])
        )
    )
    
    return base_table_4


def create_base_table_5(base_table_4):
    """
    Create fifth base table with loan status fields.
    """
    base_table_5 = base_table_4.assign(
        LAST_DTE=lambda x: np.where(x['disp_dte'] != '', x['disp_dte'], x['LAST_ACTIVITY_DATE']),
        repch_flag=lambda x: np.where(x['repch_flag'] == 'Y', 1, 0),
        PFG_COST=lambda x: x['prin_forg_upb'],
        MOD_FLAG=lambda x: np.where(x['FMOD_DTE'].notna(), 1, 0),
        MODFG_COST=lambda x: np.where(x['mod_ind'] == 'Y', 0, np.nan),
    )
    
    base_table_5 = base_table_5.assign(
        MODFG_COST=lambda x: np.where((x['mod_ind'] == 'Y') & (x['PFG_COST'] > 0), x['PFG_COST'], 0),
        MODTRM_CHNG=lambda x: np.where(x['MODTRM_CHNG'].isna(), 0, x['MODTRM_CHNG']),
        MODUPB_CHNG=lambda x: np.where(x['MODUPB_CHNG'].isna(), 0, x['MODUPB_CHNG']),
    )
    
    # Calculate CSCORE_MN in stages
    base_table_5 = base_table_5.assign(
        CSCORE_MN=lambda x: np.where((x['CSCORE_C'].notna()) & (x['CSCORE_C'] < x['CSCORE_B']), 
                                      x['CSCORE_C'], x['CSCORE_B'])
    )
    base_table_5 = base_table_5.assign(
        CSCORE_MN=lambda x: np.where(x['CSCORE_MN'].isna(), x['CSCORE_B'], x['CSCORE_MN'])
    )
    base_table_5 = base_table_5.assign(
        CSCORE_MN=lambda x: np.where(x['CSCORE_MN'].isna(), x['CSCORE_C'], x['CSCORE_MN'])
    )
    
    base_table_5 = base_table_5.assign(
        ORIG_VAL=lambda x: (x['orig_amt'] / (x['oltv'] / 100)).round(2),
        dlq_status=lambda x: np.where((x['dlq_status'] == 'X') | (x['dlq_status'] == 'XX'), 
                                       '999', x['dlq_status']),
        z_last_status=lambda x: pd.to_numeric(x['dlq_status'], errors='coerce')
    )
    
    # Create LAST_STAT using np.select
    base_table_5 = base_table_5.assign(
        LAST_STAT=lambda x: np.select(
            [
                x['zb_code'] == '09',
                x['zb_code'] == '03',
                x['zb_code'] == '02',
                x['zb_code'] == '06',
                x['zb_code'] == '15',
                x['zb_code'] == '16',
                x['zb_code'] == '01',
                (x['z_last_status'] < 999) & (x['z_last_status'] >= 9),
                x['z_last_status'] == 8,
                x['z_last_status'] == 7,
                x['z_last_status'] == 6,
                x['z_last_status'] == 5,
                x['z_last_status'] == 4,
                x['z_last_status'] == 3,
                x['z_last_status'] == 2,
                x['z_last_status'] == 1,
                x['z_last_status'] == 0
            ],
            ['F', 'S', 'T', 'R', 'N', 'L', 'P', '9', '8', '7', '6', '5', '4', '3', '2', '1', 'C'],
            default='X'
        )
    )
    
    base_table_5 = base_table_5.assign(
        FCC_DTE=lambda x: np.where(
            (x['FCC_DTE'] == '') & ((x['LAST_STAT'] == 'F') | (x['LAST_STAT'] == 'S') | 
             (x['LAST_STAT'] == 'N') | (x['LAST_STAT'] == 'T')),
            x['zb_date'], x['FCC_DTE']
        ),
        COMPLT_FLG=lambda x: np.where(x['DISP_DTE'] != '', 1, 0)
    )
    
    base_table_5 = base_table_5.assign(
        COMPLT_FLG=lambda x: np.where(
            (x['LAST_STAT'] != 'F') & (x['LAST_STAT'] != 'S') & 
            (x['LAST_STAT'] != 'N') & (x['LAST_STAT'] != 'T'),
            np.nan, x['COMPLT_FLG']
        )
    )
    
    # Calculate INT_COST
    base_table_5 = base_table_5.assign(
        INT_COST=lambda x: np.where(
            (x['COMPLT_FLG'] == 1) & (x['LPI_DTE'] != ''),
            ((x['LAST_DTE'].str[:4].astype(float) * 12 + x['LAST_DTE'].str[5:7].astype(float)) -
             (x['LPI_DTE'].str[:4].astype(float) * 12 + x['LPI_DTE'].str[5:7].astype(float))) *
            (((x['LAST_RT'] / 100) - 0.0035) / 12) * (x['LAST_UPB'] + (-1 * x['NON_INT_UPB'])),
            np.nan
        ).round(2)
    )
    
    # Fill missing costs with 0 when COMPLT_FLG == 1
    base_table_5 = base_table_5.assign(
        INT_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['INT_COST'].isna(), 0, x['INT_COST']),
        FCC_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['FCC_COST'].isna(), 0, x['FCC_COST']),
        PP_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['PP_COST'].isna(), 0, x['PP_COST']),
        AR_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['AR_COST'].isna(), 0, x['AR_COST']),
        IE_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['IE_COST'].isna(), 0, x['IE_COST']),
        TAX_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['TAX_COST'].isna(), 0, x['TAX_COST']),
        PFG_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['PFG_COST'].isna(), 0, x['PFG_COST']),
        CE_PROCS=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['CE_PROCS'].isna(), 0, x['CE_PROCS']),
        NS_PROCS=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['NS_PROCS'].isna(), 0, x['NS_PROCS']),
        RMW_PROCS=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['RMW_PROCS'].isna(), 0, x['RMW_PROCS']),
        O_PROCS=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['O_PROCS'].isna(), 0, x['O_PROCS'])
    )
    
    # Calculate NET_LOSS and NET_SEV
    base_table_5 = base_table_5.assign(
        NET_LOSS=lambda x: np.where(
            x['COMPLT_FLG'] == 1,
            (x['LAST_UPB'] + x['FCC_COST'] + x['PP_COST'] + x['AR_COST'] + x['IE_COST'] + 
             x['TAX_COST'] + x['PFG_COST'] + x['INT_COST'] + -1*x['NS_PROCS'] + 
             -1*x['CE_PROCS'] + -1*x['RMW_PROCS'] + -1*x['O_PROCS']),
            np.nan
        ).round(2),
        NET_SEV=lambda x: np.where(
            x['COMPLT_FLG'] == 1,
            x['NET_LOSS'] / x['LAST_UPB'],
            np.nan
        ).round(6)
    )
    
    return base_table_5


def create_base_table_6(base_table_5, base_table_1, performance_file):
    """
    Create sixth base table with loan modification costs.
    """
    # Calculate modification costs
    modir_table = (base_table_1
        .merge(performance_file, on='LOAN_ID', how='left')
        .query('mod_ind == "Y"')
        .assign(
            non_int_upb=lambda x: np.where(x['non_int_upb'].isna(), 0, x['non_int_upb']),
            modir_cost=lambda x: np.where(
                x['mod_ind'] == 'Y',
                (((x['orig_rt'] - x['curr_rte']) / 1200) * x['act_upb']).round(2),
                0
            ),
            modfb_cost=lambda x: np.where(
                (x['mod_ind'] == 'Y') & (x['non_int_upb'] > 0),
                ((x['curr_rte'] / 1200) * x['non_int_upb']).round(2),
                0
            )
        )
        .groupby('LOAN_ID')
        .agg(
            MODIR_COST=('modir_cost', 'sum'),
            MODFB_COST=('modfb_cost', 'sum')
        )
        .reset_index()
        .assign(MODTOT_COST=lambda x: (x['MODFB_COST'] + x['MODIR_COST']).round(2))
    )
    
    # Merge with base table 5
    base_table_6 = (base_table_5
        .merge(modir_table, on='LOAN_ID', how='left')
        .assign(COMPLT_FLG=lambda x: x['COMPLT_FLG'].astype(str))
        .assign(COMPLT_FLG=lambda x: np.where(x['COMPLT_FLG'] == 'nan', '', x['COMPLT_FLG']))
        .assign(non_int_upb=lambda x: np.where((x['COMPLT_FLG'] == '1') & x['non_int_upb'].isna(), 0, x['non_int_upb']))
    )
    
    # Calculate adjusted MODIR_COST and MODFB_COST
    base_table_6 = (
        base_table_6
        .assign(MODIR_COST=lambda x: np.where(x['COMPLT_FLG'] == '1',
            (x['MODIR_COST'] + 
             ((x['LAST_DTE'].str[:4].astype(float) * 12 + x['LAST_DTE'].str[5:7].astype(float)) -
              (x['zb_date'].str[:4].astype(float) * 12 + x['zb_date'].str[5:7].astype(float))) *
             ((x['orig_rt'] - x['LAST_RT']) / 1200) * x['LAST_UPB']).round(2),
            x['MODIR_COST']
        ))
        .assign(MODFB_COST=lambda x: np.where(
            x['COMPLT_FLG'] == '1',
            (x['MODFB_COST'] +
             ((x['LAST_DTE'].str[:4].astype(float) * 12 + x['LAST_DTE'].str[5:7].astype(float)) -
              (x['zb_date'].str[:4].astype(float) * 12 + x['zb_date'].str[5:7].astype(float))) *
             (x['LAST_RT'] / 1200) * x['non_int_upb']).round(2),
            x['MODFB_COST']
        ))
        .assign(COMPLT_FLG=lambda x: pd.to_numeric(x['COMPLT_FLG'], errors='coerce'))
        .assign(orig_rt=lambda x: x['orig_rt'].round(3))
    )
    
    return base_table_6


def create_final_output(base_table_6):
    """
    Create the final output table with selected columns.
    """
    keep_cols = [
        'LOAN_ID', 'ORIG_CHN', 'SELLER', 'orig_rt', 'orig_amt',
        'orig_trm', 'oltv', 'ocltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT',
        'occ_stat', 'state', 'zip_3', 'mi_pct', 'CSCORE_C',
        'relo_flg', 'MI_TYPE', 'AQSN_DTE', 'ORIG_DTE', 'FRST_DTE',
        'LAST_RT', 'LAST_UPB', 'msa', 'FCC_COST', 'PP_COST',
        'AR_COST', 'IE_COST', 'TAX_COST', 'NS_PROCS', 'CE_PROCS',
        'RMW_PROCS', 'O_PROCS', 'repch_flag', 'LAST_ACTIVITY_DATE',
        'LPI_DTE', 'FCC_DTE', 'DISP_DTE', 'SERVICER', 'F30_DTE',
        'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE',
        'F180_UPB', 'FCE_UPB', 'F30_UPB', 'F60_UPB', 'F90_UPB',
        'MOD_FLAG', 'FMOD_DTE', 'FMOD_UPB', 'MODIR_COST', 'MODFB_COST',
        'MODFG_COST', 'MODTRM_CHNG', 'MODUPB_CHNG', 'z_num_periods_120', 'F120_UPB',
        'CSCORE_MN', 'ORIG_VAL', 'LAST_DTE', 'LAST_STAT', 'COMPLT_FLG',
        'INT_COST', 'PFG_COST', 'NET_LOSS', 'NET_SEV', 'MODTOT_COST'
    ]
    
    final_output = base_table_6.select(columns=keep_cols)
    
    return final_output

if __name__ == "__main__":
    main()

