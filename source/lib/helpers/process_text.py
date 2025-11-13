import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import re
from datetime import datetime

def clean_text(text, normalize=True, lower=True, remove_whitespace=True, remove_html=True, remove_urls=True):
    if isinstance(text, str):
        if not text:
            return pd.NA
        
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii') if normalize else text
        text = re.sub(r'<.*?>', '', text) if remove_html else text
        text = re.sub(r'(https?://\S+|www\.\S+)', '', text) if remove_urls else text
        text = re.sub(r'[\n\t\s]+', ' ', text) if remove_whitespace else text
        text = text.lower() if lower else text
        text = text.strip()
        
        if not text:
            return pd.NA
        
        return text
    
    elif isinstance(text, pd.Series):
        text = text.fillna('')
        text = text.str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('ascii') if normalize else text
        text = text.str.replace(r'<.*?>', '', regex=True) if remove_html else text
        text = text.str.replace(r'(https?://\S+|www\.\S+)', '', regex=True) if remove_urls else text
        text = text.str.replace(r'[\n\t\s]+', ' ', regex=True) if remove_whitespace else text
        text = text.str.lower() if lower else text
        text = text.str.strip()
        text = text.replace('', pd.NA)
        return text
    
    else:
        raise TypeError("Input must be a string or a pandas Series.")


def clean_date(date, pattern=None, output_format="%Y-%m-%d", aggregation=None):
    """
    Clean and standardize date formats.
    
    Returns:
    - If aggregation='year': Period with year precision (displays as '2023')
    - If aggregation='month': Period with month precision (displays as '2023-01')
    - Otherwise: datetime64 (displays as '2023-01-15' in DataFrames)
    - If as_string=True: String formatted according to output_format
    """
    MONTH_MAP = {
        "jan": "01", "january": "01",
        "feb": "02", "february": "02",
        "mar": "03", "march": "03",
        "apr": "04", "april": "04",
        "may": "05",
        "jun": "06", "june": "06",
        "jul": "07", "july": "07",
        "aug": "08", "august": "08",
        "sep": "09", "september": "09",
        "oct": "10", "october": "10",
        "nov": "11", "november": "11",
        "dec": "12", "december": "12"
    }
    
    PATTERNS = {
        "yyyy-mm-dd": r'(\d{4})-(\d{2})-(\d{2})',
        "dd/mm/yyyy": r'(\d{1,2})/(\d{1,3})/(\d{2,4})',
        "dd/mon/yyyy": r'(\d{1,2})/([a-z]{3})/(\d{4})',
        "dd-mon-yyyy": r'(\d{1,2})-([a-z]{3})-(\d{4})',
        'month dd, yyyy': r'\s*([a-z]{3,9})\s*(\d{1,2}),\s*(\d{4})\s*',
        'mmyyyy': r'(\d{2})(\d{4})',
        'mm/yyyy': r'(\d{2})/(\d{4})',
        'mm-yyyy': r'(\d{2})-(\d{4})',
        'month, yyyy': r'([a-z]{3,9}),\s*(\d{4})',
        'yyyymm': r'(\d{4})(\d{2})'
    }
    
    if aggregation and aggregation not in ["day", "month", "year"]:
        raise ValueError("`aggregation` must be either 'day', 'month', 'year', or `None`.")
    
    if isinstance(date, pd.Series) and pd.api.types.is_datetime64_any_dtype(date):
        result = date.dt.floor('D')
        if aggregation == "year":
            return result.dt.to_period('Y')
        elif aggregation == "month":
            return result.dt.to_period('M')
        return result
    
    if isinstance(date, str):
        date = clean_text(str(date))
        if not date:
            return pd.NA
        
        if not pattern:
            matched_format = None
            for pattern_name, regex in PATTERNS.items():
                if re.match(regex, date):
                    matched_format = pattern_name
                    break
            
            if not matched_format:
                return pd.NA
        else:
            matched_format = pattern
            
        if matched_format == "yyyy-mm-dd":
            result = pd.to_datetime(date, format="%Y-%m-%d", errors='coerce')
            
        elif matched_format in ["dd/mm/yyyy", "dd/mm/yy"]:
            day, month, year = re.match(PATTERNS[matched_format], date).groups()
            result = pd.to_datetime(f"{year}-{month.zfill(2)}-{day.zfill(2)}", errors='coerce')
        
        elif matched_format in ["dd/mon/yyyy", "dd-mon-yyyy"]:
            day, month, year = re.match(PATTERNS[matched_format], date).groups()
            month = MONTH_MAP[month.lower()]
            result = pd.to_datetime(f"{year}-{month}-{day.zfill(2)}", errors='coerce')
        
        elif matched_format == "month dd, yyyy":
            month, day, year = re.match(PATTERNS[matched_format], date).groups()
            month = MONTH_MAP[month.lower()]
            result = pd.to_datetime(f"{year}-{month}-{day.zfill(2)}", errors='coerce')
        
        elif matched_format in ["mm/yyyy", "mm-yyyy", "mmyyyy"]:
            month, year = re.match(PATTERNS[matched_format], date).groups()
            result = pd.to_datetime(f"{year}-{month}-01", errors='coerce')
        
        elif matched_format == "month, yyyy":
            month, year = re.match(PATTERNS[matched_format], date).groups()
            month = MONTH_MAP[month.lower()]
            result = pd.to_datetime(f"{year}-{month}-01", errors='coerce')
        
        elif matched_format == "yyyymm":
            year, month = re.match(PATTERNS[matched_format], date).groups()
            result = pd.to_datetime(f"{year}-{month}-01", errors='coerce')
        
        else:
            return pd.NA
        
        if aggregation == "year":
            return result.to_period('Y')
        elif aggregation == "month":
            return result.to_period('M')
            
        return result
    
    elif isinstance(date, pd.Series):
        date = clean_text(date.astype("string"))
        
        test_date = None
        for val in date:
            if pd.notna(val) and val:
                test_date = val
                break
        if not pattern: 
            if not test_date:
                return pd.Series(pd.NA, index=date.index, dtype='object')
            
            matched_format = None
            for pattern_name, regex in PATTERNS.items():
                if re.match(regex, test_date):
                    matched_format = pattern_name
                    break
            
            if not matched_format:
                return pd.Series(pd.NA, index=date.index, dtype='object')
        else:
            matched_format = pattern
        
        # Parse based on pattern
        if matched_format == "yyyy-mm-dd":
            result = pd.to_datetime(date, format="%Y-%m-%d", errors='coerce')
        
        elif matched_format in ["dd/mm/yyyy", "dd/mm/yy"]:
            extracted = date.str.extract(PATTERNS[matched_format])
            extracted.columns = ["day", "month", "year"]
            result = pd.to_datetime(
                extracted["year"] + "-" + extracted["month"].str.zfill(2) + "-" + extracted["day"].str.zfill(2),
                errors="coerce"
            )
        
        elif matched_format in ["dd/mon/yyyy", "dd-mon-yyyy"]:
            extracted = date.str.extract(PATTERNS[matched_format])
            extracted.columns = ["day", "month", "year"]
            extracted["month"] = extracted["month"].str.lower().map(MONTH_MAP)
            result = pd.to_datetime(
                extracted["year"] + "-" + extracted["month"] + "-" + extracted["day"].str.zfill(2),
                errors="coerce"
            )
        
        elif matched_format == "month dd, yyyy":
            extracted = date.str.extract(PATTERNS[matched_format])
            extracted.columns = ["month", "day", "year"]
            extracted["month"] = extracted["month"].str.lower().map(MONTH_MAP)
            result = pd.to_datetime(
                extracted["year"] + "-" + extracted["month"] + "-" + extracted["day"].str.zfill(2),
                errors="coerce"
            )
        
        elif matched_format in ["mm/yyyy", "mm-yyyy", "mmyyyy"]:
            extracted = date.str.extract(PATTERNS[matched_format])
            extracted.columns = ["month", "year"]
            result = pd.to_datetime(
                extracted["year"] + "-" + extracted["month"] + "-01",
                errors="coerce"
            )
        
        elif matched_format == "month, yyyy":
            extracted = date.str.extract(PATTERNS[matched_format])
            extracted.columns = ["month", "year"]
            extracted["month"] = extracted["month"].str.lower().map(MONTH_MAP)
            result = pd.to_datetime(
                extracted["year"] + "-" + extracted["month"] + "-01",
                errors="coerce"
            )
        
        elif matched_format == "yyyymm":
            extracted = date.str.extract(PATTERNS[matched_format])
            extracted.columns = ["year", "month"]
            result = pd.to_datetime(
                extracted["year"] + "-" + extracted["month"] + "-01",
                errors="coerce"
            )
        
        else:
            return pd.Series(pd.NA, index=date.index, dtype='object')
        
        if aggregation == "year":
            return result.dt.to_period('Y')
        elif aggregation == "month":
            return result.dt.to_period('M')
        
        return result
    
    else:
        raise TypeError("Input must be a string or a pandas Series.")

