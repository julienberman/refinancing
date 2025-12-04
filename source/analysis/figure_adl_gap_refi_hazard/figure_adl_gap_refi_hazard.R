library(dplyr)
library(data.table)
library(fixest)
library(broom)
library(ggplot2)
library(arrow)

main <- function() {
    INDIR <- "datastore/output/derived/fannie_mae"
    OUTDIR <- "output/analysis/figure_adl_gap_refi_hazard"
    if (!dir.exists(OUTDIR)) {
        dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
    }

    df <- read_parquet(file.path(INDIR, "sflp_sample_processed_high.parquet"))

    model <- feols(
        exit_t1 ~ 0 + i(adl_gap_adj_bin),
        data = df
    )

    results <- tidy(model, conf.int = TRUE) %>%
        filter(grepl("adl_gap_adj_bin::", term)) %>%
        mutate(bin = as.numeric(sub("adl_gap_adj_bin::", "", term))) %>%
        filter(bin >= 9 & bin <= 30)

    figure_1 <- ggplot(results, aes(x = bin, y = estimate)) +
        geom_point(size = 3, color = "#2C3E50") +
        geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.3, linewidth = 0.8, color = "#2C3E50") +
        geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.7) +
        geom_vline(xintercept = 14, linetype = "dotted", color = "#E74C3C", linewidth = 0.7) +
        scale_x_continuous(breaks = results$bin) +
        labs(
            x = "Difference between Adjusted Rate Gap and ADL (2013) Optimal Refinancing Threshold",
            y = "Share of Loans Refinanced within 1 Month"
        ) +
        theme_minimal(base_size = 12) +
        theme(
            plot.title = element_text(face = "bold", size = 14),
            axis.text.x = element_text(angle = 45, hjust = 1),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank()
        )

    ggsave(file.path(OUTDIR, "figure_adl_gap_refi_hazard.png"), figure_1, width = 8, height = 6, dpi = 300)
}

main()
