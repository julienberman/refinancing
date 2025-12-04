from fredapi import Fred
import pandas as pd
from pathlib import Path
import json

from source.lib.helpers.process_text import clean_date
from source.lib.save_data import save_data

def main():
    OUTDIR = Path('output/derived/fred')
    with open('source/lib/api_keys.json', 'r') as f:
        API_KEYS = json.load(f)
    
    fred = Fred(api_key=API_KEYS['fred'])
    mortgage30us = (
        fred
        .get_series('MORTGAGE30US')
        .to_frame(name='mortgage_rate')
        .reset_index(names=['date'])
        .assign(date = lambda x: clean_date(x['date'], aggregation='month'))
        .groupby('date', as_index=False)
        .agg({'mortgage_rate': 'mean'})
    )

    save_data(
        mortgage30us,
        keys = ['date'],
        out_file = OUTDIR / 'mortgage30us.csv',
        log_file = OUTDIR / 'mortgage30us.log',
        sortbykey = True
    )

if __name__ == "__main__":
    main()


