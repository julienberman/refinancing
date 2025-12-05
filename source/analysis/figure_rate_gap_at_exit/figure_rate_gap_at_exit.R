library(dplyr)
library(data.table)
library(fixest)
library(broom)
library(ggplot2)
library(arrow)

main <- function() {
    INDIR <- "datastore/output/derived/fannie_mae"
    OUTDIR <- "output/analysis/figure_rate_gap_at_exit"
    if (!dir.exists(OUTDIR)) {
        dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
    }

    df <- read_parquet(file.path(INDIR, "sflp_sample_processed_high.parquet"))

    df_at_exit <- df %>% filter(period == (period_exit - 1))

    # figure_1 <- ggplot(df_at_exit, aes(x = rate_gap)) +
    #     geom_histogram(bins = 100, fill = "steelblue", color = "black") +
    #     labs(x = "Rate Gap", y = "Count") +
    #     theme_minimal()

    # ggsave(file.path(OUTDIR, "figure_rate_gap_at_exit.png"), figure_1, width = 8, height = 6, dpi = 300)

    figure_2 <- ggplot(df_at_exit, aes(x = rate_gap_adj)) +
        geom_histogram(bins = 100, fill = "steelblue", color = "black") +
        labs(x = "Adjusted Rate Gap", y = "Number of Loans") +
        theme_minimal()

    ggsave(file.path(OUTDIR, "figure_rate_gap_adj_at_exit.eps"), figure_2, width = 8, height = 6, dpi = 300)
}

main()
