library(dplyr)
library(data.table)
library(fixest)
library(broom)
library(ggplot2)

main <- function() {
    INDIR <- "datastore/output/derived/fannie_mae"

    fannie_mae <- fread(file.path(INDIR, "sflp_sample_processed_high.parquet"))

    model <- feols(
        exit_t1 ~ 0 + i(adl_gap_bin),
        data = fannie_mae
    )

    plot_refi_probability(
        model,
        xlabel = "Rate Gap - ADL Threshold",
        ylabel = "Coefficient Estimate",
        var = "adl_gap_bin",
        save = TRUE,
        out_file = "output/analysis/fannie_mae/figure_adl_gap_refi_hazard.png"
    )
}

plot_refi_probability <- function(
  model,
  title = "Effect of Rate Gap on Exit Probability",
  xlabel = "Rate Gap",
  ylabel = "Coefficient Estimate",
  save = FALSE,
  var = "adl_gap_bin",
  out_file = "output/analysis/fannie_mae/refi_probability.png"
) {
    results <- tidy(model, conf.int = TRUE) %>%
        filter(grepl(var, term)) %>%
        mutate(bin = as.numeric(sub(paste0(var, "::"), "", term)))

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
