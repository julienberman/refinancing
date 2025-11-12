import pandas as pd
from datetime import datetime

def get_quarters(start_date, end_date):
    if not start_date or not end_date:
        raise ValueError('Please provide the following parameters: `start_date`, `end_date`')
    
    start_date = pd.Period(start_date, freq='Q')
    end_date = pd.Period(end_date, freq='Q')
    quarters = pd.date_range(start=start_date, end=end_date, freq='Q')
    return [f"{q.year}Q{q.quarter}" for q in quarters]