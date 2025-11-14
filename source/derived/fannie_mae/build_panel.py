import pandas as pd



def main():
    pass

def add_event_indicators(df):
    
    indicator_map = {
        1: 'exit_t1',
        3: 'exit_t3',
        6: 'exit_t6',
        12: 'exit_t12',
        24: 'exit_t24'
    }
    
    df_with_indicators = df.copy()
    for window, indicator in indicator_map.items():
        df_with_indicators[indicator] = (
            df_with_indicators.groupby('loan_id')['exit_code']
            .transform(lambda x: x.shift(-window+1).rolling(window=window, min_periods=1).apply(lambda y: y.notna().any(), raw=True))
            .fillna(0)
            .astype(int)
        )

    return df_with_indicators


if __name__ == '__main__':
    main()

