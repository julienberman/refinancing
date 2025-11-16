import pandas as pd
import json
import os
import glob
from pathlib import Path

from source.lib.helpers.utils import get_quarters

def main():
    with open('source/lib/config.json', 'r') as f:
        CONFIG = json.load(f)

    INDIR = Path("datastore/output/derived/fannie_mae")
    OUTDIR = Path("datastore/output/derived/fannie_mae")
    START_DATE, END_DATE = CONFIG['SAMPLE_START'], CONFIG['SAMPLE_END']
    QUARTERS = get_quarters(START_DATE, END_DATE)
    CHUNKSIZE = CONFIG['CHUNKSIZE']
    
    for quarter in QUARTERS:
        print(f"Processing {quarter}...")
        df = pd.read_parquet(INDIR / "sflp_clean" / f"{quarter}.parquet")
        quarter_outdir = OUTDIR / "sflp_clean" / quarter
        quarter_outdir.mkdir(parents=True, exist_ok=True)
        
        for i in range(0, len(df), CHUNKSIZE):
            df_part = df.iloc[i: i + CHUNKSIZE]
            part = i // CHUNKSIZE
            df_part.to_parquet(quarter_outdir / f"part_{part}.parquet", index = False)
    
    # Clean up the original .parquet files
    for file in glob.glob(str(INDIR / "sflp_clean" / "*.parquet")):
        print(f"Removing {file}...")
        os.remove(file)
    
if __name__ == "__main__":
    main()

