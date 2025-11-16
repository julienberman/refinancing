library(dplyr)
library(data.table)
library(fixest)
library(broom)
library(ggplot2)

INDIR <- "datastore/output/derived/fannie_mae"

fannie_mae <- fread(file.path(INDIR, "sflp_sample.csv"))


model_1 <- feols(
    exit_t1 ~ 0 + i(rate_gap_bin),
    data = fannie_mae %>% mutate(rate_gap_bin = as.factor(rate_gap_bin))
)

# model_2 <- feols(
#     exit_t1 ~ 0 + i(rate_gap_bin),
#     data = fannie_mae %>% mutate(rate_gap_bin = as.factor(rate_gap_bin))
# )

results <- tidy(model_1, conf.int = TRUE) %>%
    filter(grepl("rate_gap_bin", term)) %>%
    mutate(bin = as.numeric(sub("rate_gap_bin::", "", term))) %>%
    arrange(bin)

ggplot(results, aes(x = bin, y = estimate)) +
    geom_point(size = 3, color = "#2C3E50") +
    geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.3, linewidth = 0.8, color = "#2C3E50") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#E74C3C", linewidth = 0.7) +
    labs(
        title = "Effect of Rate Gap on Exit Probability",
        x = "Rate Gap Bin",
        y = "Coefficient Estimate",
        caption = "95% confidence intervals shown"
    ) +
    theme_minimal(base_size = 12) +
    theme(
        plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank()
    )


# main <- function() {
#     INDIR <- "datastore/output/derived/fannie_mae"

#     fannie_mae <- fread(file.path(INDIR, "sflp_sample.csv"))

#     model_1 <- feols(
#         exit_t1 ~ i(rate_gap_bin) + time_to_maturity + credit_score_orig + upb_orig + ltv + dti + n_borrowers | period + zip,
#         data = fannie_mae
#     )


# }

# main()
