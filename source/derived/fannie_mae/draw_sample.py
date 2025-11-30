import json
from pathlib import Path
import numpy as np
import dask.dataframe as dd
from dask import compute, delayed
from dask.distributed import Client, LocalCluster

from source.lib.save_data import save_data

def main():
    with open('source/lib/config.json', 'r') as f:
        CONFIG = json.load(f)
    
    INDIR = Path('datastore/output/derived/fannie_mae')
    OUTDIR = Path('datastore/output/derived/fannie_mae')
    SEED = CONFIG['SEED']
    SAMPLE_SIZE = CONFIG['SAMPLE_SIZE']
    
    cluster = LocalCluster(n_workers=32, threads_per_worker=1, memory_limit='16 GiB')
    client = Client(cluster)

    ddf = dd.read_parquet(INDIR / 'sflp_clean')
    
    sample = build_sample(ddf, random_state=SEED, sample_size=0.001).compute()
    
    save_data(
        sample,
        keys = ['loan_id', 'period'],
        out_file = OUTDIR / "sflp_sample.parquet",
        log_file = OUTDIR / "sflp_sample.log",
        sortbykey = True
    )

# def build_sample(ddf, random_state=123, sample_size=0.01):
#     np.random.seed(random_state)
#     mask = ((ddf['mortgage_type'] == 'fixed') & (ddf['term'] == 360))
#     ddf_filtered = ddf[mask]
#     units = ddf_filtered['loan_id'].unique().compute()
#     sample_units = np.random.choice(units, size=int(sample_size * len(units)), replace=False)
#     ddf_sample = ddf_filtered[ddf_filtered['loan_id'].isin(sample_units)]
#     return ddf_sample

def build_sample(ddf, random_state=123, sample_size=0.005):
    mask = ((ddf['mortgage_type'] == 'fixed') & (ddf['term'] == 360))
    ddf_filtered = ddf[mask]
    ids = ddf_filtered[['loan_id', 'period_orig']].drop_duplicates().compute()
    sample_ids = ids.groupby('period_orig').sample(frac=sample_size, random_state=random_state)['loan_id'].tolist()
    ddf_sample = ddf_filtered[ddf_filtered['loan_id'].isin(sample_ids)]
    return ddf_sample

if __name__ == "__main__":
    main()

