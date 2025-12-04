import pandas as pd
import janitor
from pathlib import Path
import matplotlib.pyplot as plt


from source.lib.helpers.plot import plot_scatter

def main():
    INDIR_SCE = Path('output/derived/sce')
    INDIR_MORTGAGE_RATES = Path('output/derived/mortgage_rates')
    OUTDIR = Path('output/analysis')
    
    sce_housing = pd.read_csv(INDIR_SCE / 'sce_housing.csv')
    sce_credit = pd.read_csv(INDIR_SCE / 'sce_credit.csv')
    mortgage_rates = pd.read_csv(INDIR_MORTGAGE_RATES / 'mortgage30us.csv')
    print(sce_housing['mortgage_rate_type'].value_counts())
    data = (
        sce_housing
        # pd.concat([sce_housing, sce_credit])
        .merge(mortgage_rates, on='date', how='left')
        .query('residence_status_own == "yes" and mortgage_rate_on_own_home.notna()')
        .assign(likelihood_refi_1y_ahead = lambda x: x['likelihood_refi_1y_ahead'] / 100)
        .assign(likelihood_sell_home_1y_ahead = lambda x: x['likelihood_sell_home_1y_ahead'] / 100)
        .assign(likelihood_move_1y_ahead = lambda x: x['likelihood_move_1y_ahead'] / 100)
        .assign(mortgage_rate_diff = lambda x: x['mortgage_rate_on_own_home'] - x['mortgage_rate_avg_current'])
        .query('-12.5 < mortgage_rate_diff < 12.5')
        .query('mortgage_rate_type == "variable"')
        .select(columns=['sce_id', 'date', 'mortgage_rate', 'mortgage_rate_on_own_home', 'mortgage_rate_diff', 'likelihood_refi_1y_ahead', 'likelihood_sell_home_1y_ahead', 'likelihood_move_1y_ahead'])
    )
    
    plot_scatter(
        data, 
        x_var='mortgage_rate_diff', 
        y_var='likelihood_refi_1y_ahead', 
        out_file=OUTDIR / 'elasticity_refi',
        fit_line=True,
        show_equation=True,
        output_format='png'
    )
    
    plot_scatter(
        data, 
        x_var='mortgage_rate_diff', 
        y_var='likelihood_sell_home_1y_ahead', 
        out_file=OUTDIR / 'elasticity_sell_home',
        fit_line=True,
        show_equation=True,
        output_format='png'
    )
    
    plot_scatter(
        data, 
        x_var='mortgage_rate_diff', 
        y_var='likelihood_move_1y_ahead', 
        out_file=OUTDIR / 'elasticity_move',
        fit_line=True,
        show_equation=True,
        output_format='png'
    )
    
    
if __name__ == "__main__":
    main()




