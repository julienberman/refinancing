import pandas as pd
from datetime import datetime

def get_quarters(start_date, end_date):
    if not start_date or not end_date:
        raise ValueError("Please provide `start_date` and `end_date`")

    periods = pd.period_range(start=start_date, end=end_date, freq='Q')
    return [f"{p.year}Q{p.quarter}" for p in periods]

