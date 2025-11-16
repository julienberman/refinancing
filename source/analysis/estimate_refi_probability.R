library(dplyr)
library(data.table)
library(fixest)
library(arrow)

main <- function() {
    INDIR <- "datastore/output/derived/fannie_mae"

    fannie_mae <- fread(file.path(INDIR, "sflp_clean.parquet"))

    formula_1 <- as.formula()
}





main()
