import pandas as pd
from datetime import datetime

def get_quarters(start_date, end_date):
    if not start_date or not end_date:
        raise ValueError("Please provide `start_date` and `end_date`")

    periods = pd.period_range(start=start_date, end=end_date, freq='Q')
    return [f"{p.year}Q{p.quarter}" for p in periods]

def relocate(df, columns, before=None, after=None):
    """
    Relocate columns in a DataFrame before or after a reference column.
    """
    if before is not None and after is not None:
        raise ValueError("Cannot specify both 'before' and 'after'")
    
    if before is None and after is None:
        raise ValueError("Must specify either 'before' or 'after'")
    if isinstance(columns, str):
        columns = [columns]
    
    all_cols = df.columns.tolist()
    remaining_cols = [col for col in all_cols if col not in columns]
    
    if before is not None:
        if before not in remaining_cols:
            raise ValueError(f"Column '{before}' not found in DataFrame")
        idx = remaining_cols.index(before)
        new_cols = remaining_cols[:idx] + columns + remaining_cols[idx:]
    else:
        if after not in remaining_cols:
            raise ValueError(f"Column '{after}' not found in DataFrame")
        idx = remaining_cols.index(after)
        new_cols = remaining_cols[:idx+1] + columns + remaining_cols[idx+1:]
    
    return df[new_cols]

@pd.api.extensions.register_dataframe_accessor("relocate")
class RelocateAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def __call__(self, columns, before=None, after=None):
        return relocate(self._obj, columns, before, after)

