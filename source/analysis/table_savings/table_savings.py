import pandas as pd
import json
from pathlib import Path

def main():
    with open('source/lib/parameters.json', 'r') as f:
        PARAMETER_LIST = json.load(f)
    
    INDIR = Path('datastore/output/derived/fannie_mae')
    OUTDIR = Path('output/analysis/fannie_mae/table_savings')
    
    autofill_list = []
    for PARAMETER_TYPE, PARAMETERS in PARAMETER_LIST.items():
        annual_discount_rate = PARAMETERS['ANNUAL_DISCOUNT_RATE']
        prob_move = PARAMETERS['PROB_MOVE']
        marginal_tax_rate = PARAMETERS['MARGINAL_TAX_RATE']
        
        df_full = pd.read_parquet(INDIR / f'sflp_sample_processed_{PARAMETER_TYPE}_full.parquet')
        df_full_aggregated = df_full.drop_duplicates(subset=['loan_id'])
        mean_savings_optimal_refi_full = df_full_aggregated['savings_optimal_refi_adj'].mean()
        mean_savings_realized_refi_full = df_full_aggregated['savings_realized_refi_adj'].mean()
        mean_savings_loss_full = df_full_aggregated['savings_loss_adj'].mean()
        autofill_list.append([PARAMETER_TYPE, "Full", annual_discount_rate, prob_move, marginal_tax_rate, mean_savings_optimal_refi_full, mean_savings_realized_refi_full, mean_savings_loss_full])
        
        df_refi_eligible = pd.read_parquet(INDIR / f'sflp_sample_processed_{PARAMETER_TYPE}_refi_eligible.parquet')
        df_refi_eligible_aggregated = df_refi_eligible.drop_duplicates(subset=['loan_id'])
        mean_savings_optimal_refi_refi_eligible = df_refi_eligible_aggregated['savings_optimal_refi_adj'].mean()
        mean_savings_realized_refi_refi_eligible = df_refi_eligible_aggregated['savings_realized_refi_adj'].mean()
        mean_savings_loss_refi_eligible = df_refi_eligible_aggregated['savings_loss_adj'].mean()
        autofill_list.append([PARAMETER_TYPE, "Refi Eligible", annual_discount_rate, prob_move, marginal_tax_rate, mean_savings_optimal_refi_refi_eligible, mean_savings_realized_refi_refi_eligible, mean_savings_loss_refi_eligible])
    
    with open(OUTDIR / 'table_savings.txt', 'w') as f:
        f.write("<tab:table_savings>")
        for row in autofill_list:
            f.write('\t'.join([str(x) for x in row]) + '\n')
    
if __name__ == '__main__':
    main()

