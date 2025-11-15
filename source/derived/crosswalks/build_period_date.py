import pandas as pd
import json
from pathlib import Path

from source.lib.save_data import save_data

def main():
    with open('source/lib/config.json', 'r') as file:
        CONFIG = json.load(file)
    PERIOD = CONFIG['PERIOD']
    OUTDIR = Path("datastore/raw/crosswalks/data")
    START_DATE = '2000-01-01'
    END_DATE = '2026-01-01'
    months = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    periods = pd.date_range(start=START_DATE, end=END_DATE, freq=PERIOD)
    
    df = pd.DataFrame({'date': months})
    df['period'] = df['date'].apply(lambda d: (periods <= d).sum())
        
    save_data(
        df,
        keys=['date'],
        out_file=OUTDIR / "cw_period_date.csv", 
        log_file=OUTDIR / "cw_period_date.log", 
        sortbykey=True
    )

if __name__ == "__main__":
    main()

