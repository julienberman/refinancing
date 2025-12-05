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
            purpose = first(purpose),
            rate_mortgage30us_orig = first(rate_mortgage30us_orig),
            period_orig = first(period_orig)
        ) %>%
        ungroup() %>%
        mutate(
            ind_refi_cash = if_else(purpose == "C", 1, 0),
            ind_refi_no_cash = if_else(purpose == "R", 1, 0),
            ind_refi_unknown = if_else(purpose == "U", 1, 0)
        ) %>%
        filter(ind_refi_cash == 1 | ind_refi_no_cash == 1 | ind_refi_unknown == 1) %>%
        group_by(period_orig) %>%
        summarise(
            rate_mortgage30us_orig = first(rate_mortgage30us_orig),
            rate_refi_cash = sum(ind_refi_cash) / n(),
            rate_refi_no_cash = sum(ind_refi_no_cash) / n(),
            count = n()
        ) %>%
        ungroup() %>%
        filter(count >= 40)

    figure_1 <- ggplot(df_agg, aes(x = rate_mortgage30us_orig, y = rate_refi_cash)) +
        geom_point(size = 3, color = "#2C3E50") +
        labs(
            x = "Safe 30Y Mortgage Rate",
            y = "Share of Refinances with Cash-Out"
        ) +
        theme_minimal(base_size = 12) +
        geom_hline(yintercept = 0.2, linetype = "solid", color = "black", linewidth = 0.7) +
        geom_vline(xintercept = 2, linetype = "solid", color = "black", linewidth = 0.7)

    ggsave(file.path(OUTDIR, "figure_cash_out_share.png"), figure_1, width = 8, height = 6, dpi = 300)
}

main()
