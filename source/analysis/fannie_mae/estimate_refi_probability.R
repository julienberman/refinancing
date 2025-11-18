library(dplyr)
library(data.table)
library(fixest)
library(broom)
library(ggplot2)

main <- function() {
    INDIR <- "datastore/output/derived/fannie_mae"

    fannie_mae <- fread(file.path(INDIR, "sflp_sample.csv"))

    model_t1 <- feols(
        exit_t1 ~ 0 + i(rate_gap_bin),
        data = fannie_mae
    )

    plot_refi_probability(
        model_t1,
        title = "Effect of Rate Gap on Exit Probability: 1-month horizon",
        xlabel = "Rate Gap",
        ylabel = "Coefficient Estimate",
        save = TRUE,
        out_file = "output/analysis/fannie_mae/refi_probability_t1.png"
    )

    model_t3 <- feols(
        exit_t3 ~ 0 + i(rate_gap_bin),
        data = fannie_mae
    )

    plot_refi_probability(
        model_t3,
        title = "Effect of Rate Gap on Exit Probability: 3-month horizon",
        xlabel = "Rate Gap",
        ylabel = "Coefficient Estimate",
        save = TRUE,
        out_file = "output/analysis/fannie_mae/refi_probability_t3.png"
    )

    model_t6 <- feols(
        exit_t6 ~ 0 + i(rate_gap_bin),
        data = fannie_mae
    )

    plot_refi_probability(
        model_t6,
        title = "Effect of Rate Gap on Exit Probability: 6-month horizon",
        xlabel = "Rate Gap",
        ylabel = "Coefficient Estimate",
        save = TRUE,
        out_file = "output/analysis/fannie_mae/refi_probability_t6.png"
    )

    model_t12 <- feols(
        exit_t12 ~ 0 + i(rate_gap_bin),
        data = fannie_mae
    )

    plot_refi_probability(
        model_t12,
        title = "Effect of Rate Gap on Exit Probability: 12-month horizon",
        xlabel = "Rate Gap",
        ylabel = "Coefficient Estimate",
        save = TRUE,
        out_file = "output/analysis/fannie_mae/refi_probability_t12.png"
    )

    model_t24 <- feols(
        exit_t24 ~ 0 + i(rate_gap_bin),
        data = fannie_mae
    )

    plot_refi_probability(
        model_t24,
        title = "Effect of Rate Gap on Exit Probability: 24-month horizon",
        xlabel = "Rate Gap",
        ylabel = "Coefficient Estimate",
        save = TRUE,
        out_file = "output/analysis/fannie_mae/refi_probability_t24.png"
    )
}

plot_refi_probability <- function(
  model,
  title = "Effect of Rate Gap on Exit Probability",
  xlabel = "Rate Gap",
  ylabel = "Coefficient Estimate",
  save = FALSE,
  out_file = "output/analysis/fannie_mae/refi_probability.png"
) {
    results <- tidy(model, conf.int = TRUE) %>%
        filter(grepl("rate_gap_bin", term)) %>%
        mutate(bin = as.numeric(sub("rate_gap_bin::", "", term))) %>%
        arrange(bin) %>%
        filter(bin >= 15 & bin <= 45)

    plot <- ggplot(results, aes(x = bin, y = estimate)) +
        geom_point(size = 3, color = "#2C3E50") +
        geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.3, linewidth = 0.8, color = "#2C3E50") +
        geom_hline(yintercept = 0, linetype = "dashed", color = "#E74C3C", linewidth = 0.7) +
        geom_vline(xintercept = 31, linetype = "dotted", color = "#E74C3C", linewidth = 0.7) +
        scale_x_continuous(breaks = results$bin) +
        labs(
            title = title,
            x = xlabel,
            y = ylabel
        ) +
        theme_minimal(base_size = 12) +
        theme(
            plot.title = element_text(face = "bold", size = 14),
            axis.text.x = element_text(angle = 45, hjust = 1),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank()
        )

    if (save) {
        dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
        ggsave(out_file, plot, width = 10, height = 6, dpi = 300)
    }

    return(plot)
}

main()
