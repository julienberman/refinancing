library(dplyr)
library(data.table)
library(fixest)
library(broom)
library(ggplot2)
library(arrow)

main <- function() {
    INDIR <- "datastore/output/derived/fannie_mae"
    OUTDIR <- "output/analysis/figure_refi_delay"
    if (!dir.exists(OUTDIR)) {
        dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
    }

    df <- read_parquet(file.path(INDIR, "sflp_sample_processed_high.parquet"))

    df_longest <- df %>%
        arrange(loan_id, period) %>%
        group_by(loan_id) %>%
        mutate(
            streak_id = cumsum(should_refi == 1 & lag(should_refi, default = 0) == 0)
        ) %>%
        group_by(loan_id, streak_id) %>%
        summarise(streak_len = sum(should_refi), .groups = "drop_last") %>%
        summarise(longest_streak = max(streak_len), .groups = "drop") %>%
        filter(longest_streak != 0)

    figure_1 <- ggplot(df_longest, aes(x = longest_streak)) +
        geom_histogram(bins = 100, fill = "steelblue", color = "black") +
        labs(x = "Number of Consecutive Months not Refinancing when Optimal", y = "Number of Loans") +
        theme_minimal()

    ggsave(file.path(OUTDIR, "figure_refi_delay.eps"), figure_1, width = 8, height = 6)
}

main()
