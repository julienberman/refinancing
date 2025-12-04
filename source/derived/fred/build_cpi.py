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
    cpi = (
        fred
        .get_series('CPIAUCSL')
        .to_frame(name='cpi')
        .reset_index(names=['date'])
        .assign(date = lambda x: clean_date(x['date'], aggregation='month'))
        .groupby('date', as_index=False)
        .agg({'cpi': 'mean'})
    )

    save_data(
        cpi,
        keys = ['date'],
        out_file = OUTDIR / 'cpiaucsl.csv',
        log_file = OUTDIR / 'cpiaucsl.log',
        sortbykey = True
    )

if __name__ == "__main__":
    main()

