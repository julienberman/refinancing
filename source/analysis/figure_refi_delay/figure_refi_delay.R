library(dplyr)
library(data.table)
library(fixest)
library(broom)
library(ggplot2)

main <- function() {
    INDIR <- "datastore/output/derived/fannie_mae"

    df <- fread(file.path(INDIR, "sflp_sample_processed_high.parquet"))

    df_longest <- df %>%
        arrange(loan_id, period) %>%
        group_by(loan_id) %>%
        mutate(
            streak_id = cumsum(should_refi == 1 & lag(should_refi, default = 0) == 0)
        ) %>%
        group_by(loan_id, streak_id) %>%
        summarise(streak_len = sum(should_refi), .groups = "drop_last") %>%
        summarise(longest_streak = max(streak_len), .groups = "drop")

    figure_1 <- ggplot(df_longest, aes(x = longest_streak)) +
        geom_histogram(bins = 100, fill = "steelblue", color = "black") +
        labs(x = "Longest Streak of Consecutive Days", y = "Count") +
        theme_minimal()

    ggsave(figure_1, file.path(OUTDIR, "figure_refi_delay.png"), width = 8, height = 6)
}

main()
