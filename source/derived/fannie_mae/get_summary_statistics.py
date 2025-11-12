import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

from source.lib.helpers.utils import get_quarters
from source.lib.save_data import save_data


def main():
    INDIR = Path('datastore/raw/fannie_mae/data')
    OUTDIR = Path('output/derived/fannie_mae')
    
    fannie_mae = pd.read_parquet(INDIR / 'fannie_mae.parquet')
    fannie_mae = add_derived_fields(fannie_mae)

    disp_max_dte = pd.to_datetime(fannie_mae['AQSN_DTE'].max()) - relativedelta(months=3)
    
    aquisition_stats = generate_acquisition_stats(fannie_mae)
    performance_stats = generate_performance_stats(fannie_mae)
    disposition_loss_stats = generate_disposition_loss_stats(fannie_mae, disp_max_dte)
    origination_loss_stats = generate_origination_loss_stats(fannie_mae, disp_max_dte)
    
    save_data(
        aquisition_stats,
        keys = ...,
        out_file = OUTDIR / 'aquisition_stats.csv',
        log_file = OUTDIR / 'aquisition_stats.log',
        sortbykey = False
    )
    
    save_data(
        performance_stats,
        keys = ...,
        out_file = OUTDIR / 'performance_stats.csv',
        log_file = OUTDIR / 'performance_stats.log',
        sortbykey = False
    )
    
    save_data(
        disposition_loss_stats,
        keys = ...,
        out_file = OUTDIR / 'disposition_loss_stats.csv',
        log_file = OUTDIR / 'disposition_loss_stats.log',
        sortbykey = False
    )
        
    save_data(
        origination_loss_stats,
        keys = ...,
        out_file = OUTDIR / 'origination_loss_stats.csv',
        log_file = OUTDIR / 'origination_loss_stats.log',
        sortbykey = False
    )
    

def add_derived_fields(df):
    """
    Add derived fields needed for summary statistics.
    """
    df = df.assign(
        ORIG_YR=lambda x: x['ORIG_DTE'].str[:4],
        DISP_YR=lambda x: x['DISP_DTE'].str[:4],
        COMPLT_FLG=lambda x: pd.to_numeric(x['COMPLT_FLG'], errors='coerce'),
    )
    
    df = df.assign(
        COMPLT_FLG=lambda x: np.where(x['COMPLT_FLG'] != 1, 0, 1),
        DFLT_UPB=lambda x: x['LAST_UPB'] * x['COMPLT_FLG'],
        TOT_COST=lambda x: (x['FCC_COST'] + x['PP_COST'] + x['AR_COST'] + 
                            x['IE_COST'] + x['TAX_COST'] + x['DFLT_UPB'] + x['INT_COST']),
        TOT_PROCS=lambda x: (x['NS_PROCS'] + x['CE_PROCS'] + x['RMW_PROCS'] + x['O_PROCS']),
        LIQ_EXP=lambda x: (x['FCC_COST'] + x['PP_COST'] + x['AR_COST'] + 
                           x['IE_COST'] + x['TAX_COST'])
    )
    
    return df


def generate_acquisition_stats(df):
    """
    Generate acquisition profile statistics by origination year.
    """
    # Calculate weighted averages by year
    aqsn_stat = (
        df
        .groupby('ORIG_YR')
        .apply(lambda x: pd.Series({
            'LOAN_COUNT': len(x),
            'SUM_orig_amt': x['orig_amt'].sum() / 1_000_000,
            'AVG_orig_amt': x['orig_amt'].mean(),
            'CSCORE_B': (x['CSCORE_B'] * x['orig_amt']).sum() / x.loc[x['CSCORE_B'].notna(), 'orig_amt'].sum(),
            'CSCORE_C': (x['CSCORE_C'] * x['orig_amt']).sum() / x.loc[x['CSCORE_C'].notna(), 'orig_amt'].sum(),
            'oltv': (x['oltv'] * x['orig_amt']).sum() / x.loc[x['oltv'].notna(), 'orig_amt'].sum(),
            'ocltv': (x['ocltv'] * x['orig_amt']).sum() / x.loc[x['ocltv'].notna(), 'orig_amt'].sum(),
            'dti': (x['dti'] * x['orig_amt']).sum() / x.loc[x['dti'].notna(), 'orig_amt'].sum(),
            'orig_rt': (x['orig_rt'] * x['orig_amt']).sum() / x.loc[x['orig_rt'].notna(), 'orig_amt'].sum()
        }), include_groups=False)
        .reset_index()
    )
    
    # Calculate totals (weighted by year-level weighted averages)
    yearly_wm = (
        df
        .groupby('ORIG_YR')
        .apply(lambda x: pd.Series({
            'orig_amt': x['orig_amt'].sum(),
            'wm_CSCORE_B': (x['CSCORE_B'] * x['orig_amt']).sum() / x.loc[x['CSCORE_B'].notna(), 'orig_amt'].sum(),
            'wm_CSCORE_C': (x['CSCORE_C'] * x['orig_amt']).sum() / x.loc[x['CSCORE_C'].notna(), 'orig_amt'].sum(),
            'wm_oltv': (x['oltv'] * x['orig_amt']).sum() / x.loc[x['oltv'].notna(), 'orig_amt'].sum(),
            'wm_ocltv': (x['ocltv'] * x['orig_amt']).sum() / x.loc[x['ocltv'].notna(), 'orig_amt'].sum(),
            'wm_dti': (x['dti'] * x['orig_amt']).sum() / x.loc[x['dti'].notna(), 'orig_amt'].sum(),
            'wm_orig_rt': (x['orig_rt'] * x['orig_amt']).sum() / x.loc[x['orig_rt'].notna(), 'orig_amt'].sum()
        }), include_groups=False)
        .reset_index()
    )
    
    total_orig_amt = yearly_wm['orig_amt'].sum()
    
    aqsn_total = pd.DataFrame({
        'ORIG_YR': ['Total'],
        'LOAN_COUNT': [len(df)],
        'SUM_orig_amt': [df['orig_amt'].sum() / 1_000_000],
        'AVG_orig_amt': [df['orig_amt'].mean()],
        'CSCORE_B': [(yearly_wm['wm_CSCORE_B'] * yearly_wm['orig_amt']).sum() / total_orig_amt],
        'CSCORE_C': [(yearly_wm['wm_CSCORE_C'] * yearly_wm['orig_amt']).sum() / total_orig_amt],
        'oltv': [(yearly_wm['wm_oltv'] * yearly_wm['orig_amt']).sum() / total_orig_amt],
        'ocltv': [(yearly_wm['wm_ocltv'] * yearly_wm['orig_amt']).sum() / total_orig_amt],
        'dti': [(yearly_wm['wm_dti'] * yearly_wm['orig_amt']).sum() / total_orig_amt],
        'orig_rt': [(yearly_wm['wm_orig_rt'] * yearly_wm['orig_amt']).sum() / total_orig_amt]
    })
    
    # Combine and rename columns
    aqsn_stat = pd.concat([aqsn_stat, aqsn_total], ignore_index=True)
    
    aqsn_stat.columns = [
        "Origination Year", "Loan Count", "Total Orig. UPB ($M)", "Avg. Orig UPB ($M)",
        "Borrower Credit Score", "Co-Borrower Credit Score", "LTV Ratio", 
        "CLTV Ratio", "DTI", "Note Rate"
    ]
    
    return aqsn_stat


def generate_performance_stats(df):
    """
    Generate performance profile statistics by origination year.
    """
    # Calculate by year
    perf_stat = (df
        .groupby('ORIG_YR')
        .apply(lambda x: pd.Series({
            'LOAN_COUNT': len(x),
            'SUM_orig_amt': x['orig_amt'].sum() / 1_000_000,
            'ACTIVE_COUNT': (~x['LAST_STAT'].isin(['P', 'R', 'N', 'F', 'S', 'T', 'L'])).sum(),
            'ACTIVE_UPB': (x.loc[~x['LAST_STAT'].isin(['P', 'R', 'N', 'F', 'S', 'T', 'L']), 'LAST_UPB'].sum() / 1_000_000),
            'PREPAY': (x['LAST_STAT'] == 'P').sum(),
            'RPCH': (x['LAST_STAT'] == 'R').sum(),
            'SS': (x['LAST_STAT'] == 'S').sum(),
            'TPS': (x['LAST_STAT'] == 'T').sum(),
            'REO': (x['LAST_STAT'] == 'F').sum(),
            'NPL': (x['LAST_STAT'] == 'N').sum(),
            'RPL': (x['LAST_STAT'] == 'L').sum(),
            'MOD': x['FMOD_DTE'].notna().sum(),
            'D180_UPB': x['F180_UPB'].sum() / 1_000_000,
            'FCE_RT': x['F180_UPB'].sum() / x['orig_amt'].sum(),
            'DFLT_UPB': x['DFLT_UPB'].sum() / 1_000_000,
            'NET_LOSS_RT': x['NET_LOSS'].sum() / x['orig_amt'].sum()
        }), include_groups=False)
        .reset_index()
    )
    
    # Calculate totals
    perf_total = pd.DataFrame({
        'ORIG_YR': ['Total'],
        'LOAN_COUNT': [len(df)],
        'SUM_orig_amt': [df['orig_amt'].sum() / 1_000_000],
        'ACTIVE_COUNT': [(~df['LAST_STAT'].isin(['P', 'R', 'N', 'F', 'S', 'T', 'L'])).sum()],
        'ACTIVE_UPB': [df.loc[~df['LAST_STAT'].isin(['P', 'R', 'N', 'F', 'S', 'T', 'L']), 'LAST_UPB'].sum() / 1_000_000],
        'PREPAY': [(df['LAST_STAT'] == 'P').sum()],
        'RPCH': [(df['LAST_STAT'] == 'R').sum()],
        'SS': [(df['LAST_STAT'] == 'S').sum()],
        'TPS': [(df['LAST_STAT'] == 'T').sum()],
        'REO': [(df['LAST_STAT'] == 'F').sum()],
        'NPL': [(df['LAST_STAT'] == 'N').sum()],
        'RPL': [(df['LAST_STAT'] == 'L').sum()],
        'MOD': [df['FMOD_DTE'].notna().sum()],
        'D180_UPB': [df['F180_UPB'].sum() / 1_000_000],
        'FCE_RT': [df['F180_UPB'].sum() / df['orig_amt'].sum()],
        'DFLT_UPB': [df['DFLT_UPB'].sum() / 1_000_000],
        'NET_LOSS_RT': [df['NET_LOSS'].sum() / df['orig_amt'].sum()]
    })
    
    # Combine and rename columns
    perf_stat = pd.concat([perf_stat, perf_total], ignore_index=True)
    
    perf_stat.columns = [
        "Origination Year", "Loan Count", "Total Orig. UPB ($M)", "Loan Count (Active)",
        "Active UPB ($M)", "Prepaid (01)", "Repurchased (06)", "Short Sale (03)", 
        "Third Party Sale (02)", "REO (09)", "Non-Performing Loan Sale (15)", 
        "Re-Performing Loan Sale (16)", "Mod Loan Count", "D180 UPB",
        "D180% of Orig. UPB", "Default UPB", "Loss Rate (%)"
    ]
    
    return perf_stat


def generate_disposition_loss_stats(df, disp_max_dte):
    """
    Generate disposition loss statistics by disposition year.
    """
    # Filter and aggregate early years
    df_filtered = df[
        (df['DISP_DTE'] == '') | 
        (pd.to_datetime(df['DISP_DTE']) <= disp_max_dte)
    ].copy()
    
    df_filtered['DISP_YR'] = df_filtered['DISP_YR'].replace({
        '2000': '2000-2006', '2001': '2000-2006', '2002': '2000-2006',
        '2003': '2000-2006', '2004': '2000-2006', '2005': '2000-2006',
        '2006': '2000-2006'
    })
    
    # Calculate by disposition year
    disp_loss = (
        df_filtered
        .groupby('DISP_YR')
        .apply(lambda x: pd.Series({
            'DFLT_UPB_SUM': x['DFLT_UPB'].sum() / 1_000_000,
            'INT_COST': x['INT_COST'].sum() / x['DFLT_UPB'].sum(),
            'LIQ_EXP': x['LIQ_EXP'].sum() / x['DFLT_UPB'].sum(),
            'FCC_COST': x['FCC_COST'].sum() / x['DFLT_UPB'].sum(),
            'PP_COST': x['PP_COST'].sum() / x['DFLT_UPB'].sum(),
            'AR_COST': x['AR_COST'].sum() / x['DFLT_UPB'].sum(),
            'IE_COST': x['IE_COST'].sum() / x['DFLT_UPB'].sum(),
            'TAX_COST': x['TAX_COST'].sum() / x['DFLT_UPB'].sum(),
            'TOT_COST': x['TOT_COST'].sum() / x['DFLT_UPB'].sum(),
            'NS_PROCS': x['NS_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'CE_PROCS': x['CE_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'RMW_PROCS': x['RMW_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'O_PROCS': x['O_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'TOT_PROCS': x['TOT_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'NET_SEV': x['NET_LOSS'].sum() / x['DFLT_UPB'].sum(),
            'NET_LOSS': x['NET_LOSS'].sum() / 1_000_000
        }), include_groups=False)
        .reset_index()
    )
    
    # Calculate totals
    disp_total = pd.DataFrame({
        'DISP_YR': ['Total'],
        'DFLT_UPB_SUM': [df_filtered['DFLT_UPB'].sum() / 1_000_000],
        'INT_COST': [df_filtered['INT_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'LIQ_EXP': [df_filtered['LIQ_EXP'].sum() / df_filtered['DFLT_UPB'].sum()],
        'FCC_COST': [df_filtered['FCC_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'PP_COST': [df_filtered['PP_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'AR_COST': [df_filtered['AR_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'IE_COST': [df_filtered['IE_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'TAX_COST': [df_filtered['TAX_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'TOT_COST': [df_filtered['TOT_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'NS_PROCS': [df_filtered['NS_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'CE_PROCS': [df_filtered['CE_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'RMW_PROCS': [df_filtered['RMW_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'O_PROCS': [df_filtered['O_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'TOT_PROCS': [df_filtered['TOT_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'NET_SEV': [df_filtered['NET_LOSS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'NET_LOSS': [df_filtered['NET_LOSS'].sum() / 1_000_000]
    })
    
    disp_loss = pd.concat([disp_loss, disp_total], ignore_index=True)
    
    return disp_loss


def generate_origination_loss_stats(df, disp_max_dte):
    """
    Generate origination loss statistics by origination year.
    """
    # Filter and aggregate early years
    df_filtered = df[
        (df['DISP_DTE'] == '') | 
        (pd.to_datetime(df['DISP_DTE']) <= disp_max_dte)
    ].copy()
    
    df_filtered['ORIG_YR'] = df_filtered['ORIG_YR'].replace({
        '1999': '1999-2005', '2000': '1999-2005', '2001': '1999-2005',
        '2002': '1999-2005', '2003': '1999-2005', '2004': '1999-2005',
        '2005': '1999-2005'
    })
    
    # Calculate by origination year
    orig_loss = (
        df_filtered
        .groupby('ORIG_YR')
        .apply(lambda x: pd.Series({
            'DFLT_UPB_SUM': x['DFLT_UPB'].sum() / 1_000_000,
            'DLFT_RT': x['DFLT_UPB'].sum() / x['orig_amt'].sum(),
            'INT_COST': x['INT_COST'].sum() / x['DFLT_UPB'].sum(),
            'LIQ_EXP': x['LIQ_EXP'].sum() / x['DFLT_UPB'].sum(),
            'FCC_COST': x['FCC_COST'].sum() / x['DFLT_UPB'].sum(),
            'PP_COST': x['PP_COST'].sum() / x['DFLT_UPB'].sum(),
            'AR_COST': x['AR_COST'].sum() / x['DFLT_UPB'].sum(),
            'IE_COST': x['IE_COST'].sum() / x['DFLT_UPB'].sum(),
            'TAX_COST': x['TAX_COST'].sum() / x['DFLT_UPB'].sum(),
            'TOT_COST': x['TOT_COST'].sum() / x['DFLT_UPB'].sum(),
            'NS_PROCS': x['NS_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'CE_PROCS': x['CE_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'RMW_PROCS': x['RMW_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'O_PROCS': x['O_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'TOT_PROCS': x['TOT_PROCS'].sum() / x['DFLT_UPB'].sum(),
            'NET_SEV': x['NET_LOSS'].sum() / x['DFLT_UPB'].sum(),
            'NET_LOSS': x['NET_LOSS'].sum() / 1_000_000
        }), include_groups=False)
        .reset_index()
    )
    
    # Calculate totals
    orig_total = pd.DataFrame({
        'ORIG_YR': ['Total'],
        'DFLT_UPB_SUM': [df_filtered['DFLT_UPB'].sum() / 1_000_000],
        'DLFT_RT': [df_filtered['DFLT_UPB'].sum() / df_filtered['orig_amt'].sum()],
        'INT_COST': [df_filtered['INT_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'LIQ_EXP': [df_filtered['LIQ_EXP'].sum() / df_filtered['DFLT_UPB'].sum()],
        'FCC_COST': [df_filtered['FCC_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'PP_COST': [df_filtered['PP_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'AR_COST': [df_filtered['AR_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'IE_COST': [df_filtered['IE_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'TAX_COST': [df_filtered['TAX_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'TOT_COST': [df_filtered['TOT_COST'].sum() / df_filtered['DFLT_UPB'].sum()],
        'NS_PROCS': [df_filtered['NS_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'CE_PROCS': [df_filtered['CE_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'RMW_PROCS': [df_filtered['RMW_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'O_PROCS': [df_filtered['O_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'TOT_PROCS': [df_filtered['TOT_PROCS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'NET_SEV': [df_filtered['NET_LOSS'].sum() / df_filtered['DFLT_UPB'].sum()],
        'NET_LOSS': [df_filtered['NET_LOSS'].sum() / 1_000_000]
    })
    
    orig_loss = pd.concat([orig_loss, orig_total], ignore_index=True)

    return orig_loss

if __name__ == "__main__":
    main()


