import pandas as pd
import json
import os
import glob
from pathlib import Path

from source.lib.helpers.utils import get_quarters

def main():
    
    with open('source/lib/schemas.json', 'r') as f:
        SCHEMAS = json.load(f)
    with open('source/lib/config.json', 'r') as f:
        CONFIG = json.load(f)

    INDIR = Path("datastore/raw/fannie_mae/data")
    OUTDIR = Path("datastore/raw/fannie_mae/data")
    START_DATE, END_DATE = CONFIG['SAMPLE_START'], CONFIG['SAMPLE_END']
    QUARTERS = get_quarters(START_DATE, END_DATE)
    CHUNKSIZE = CONFIG['CHUNKSIZE']
    
    for quarter in QUARTERS:
        print(f"Processing {quarter}...")
        df = pd.read_csv(
            INDIR / f"{quarter}.csv",
            sep='|',
            names=SCHEMAS['fannie_mae'].keys(), 
            dtype=SCHEMAS['fannie_mae'], 
            low_memory=False
        )
        
        quarter_outdir = OUTDIR / quarter
        quarter_outdir.mkdir(parents=True, exist_ok=True)
        
        for i in range(0, len(df), CHUNKSIZE):
            df_part = df.iloc[i: i + CHUNKSIZE]
            part = i // CHUNKSIZE
            df_part.to_parquet(quarter_outdir / f"part_{part}.parquet", index = False)
        
        # Clean up the original .csv file
        print(f"Removing {quarter}.csv...")
        os.remove(INDIR / f"{quarter}.csv")
    
if __name__ == "__main__":
    main()

