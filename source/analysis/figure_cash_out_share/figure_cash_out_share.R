library(dplyr)
library(data.table)
library(fixest)
library(broom)
library(ggplot2)
library(binsreg)
library(arrow)

main <- function() {
    INDIR <- "datastore/output/derived/fannie_mae"
    OUTDIR <- "output/analysis/figure_cash_out_share"
    if (!dir.exists(OUTDIR)) {
        dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
    }

    df <- read_parquet(file.path(INDIR, "sflp_sample_processed_high.parquet"))

    df_agg <- df %>%
        group_by(loan_id) %>%
        summarise(
            purpose = first(purpose)
            rate_mortgage30us_orig = first(rate_mortgage30us_orig)
        ) %>% 
        ungroup() %>% 
        mutate(
            ind_refi_cash = if_else(purpose == 'refi_cash', 1, 0)
            ind_refi_no_cash = if_else(purpose == 'refi_no_cash', 1, 0)
            ind_refi_unknown = if_else(purpose == 'refi_unknown', 1, 0)
        ) %>% 
        filter(ind_refi_cash == 1 | ind_refi_no_cash == 1 | ind_refi_unknown == 1)
    
    figure_1 <- binsreg(y = 'ind_refi_cash', x = 'rate_mortgage30us_orig', data = df_agg, dots = TRUE, polyreg = 3, ci = TRUE)$bins_plot +
        labs(x = "Safe mortgage rate", y = "Share of refinances with cash-out") +
        theme_minimal()

    ggsave(file.path(OUTDIR, "figure_cash_out_share.png"), figure_1, width = 8, height = 6)
}

main()

