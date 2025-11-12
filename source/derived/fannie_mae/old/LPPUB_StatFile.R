####################################################################
# This code aggregates the Loan Performance data into a one-row-per-loan analytical dataset.
# Files are sequentially read in, modified, and then exported to .csv format in your working directory.
# We recommend running this code and LPPUB_StatSummary.R to ensure you have downloaded the data in full.
# Please be sure this code, along with LPPUB_StatFile_Production.R, is in the same directory as the Loan Performance data.
####################################################################

### Required packages
# Tested with data.table v.1.14.8 and dplyr v1.1.4
if (!(require(data.table))) install.packages("data.table")
if (!(require(dplyr))) install.packages("dplyr")

### Set up a function to read in the Loan Performance files
load_lppub_file <- function(filename, col_names, col_classes) {
    df <- fread(filename, sep = "|", col.names = col_names, colClasses = col_classes)
    return(df)
}

### DEFINE INPUT AND OUTPUT DIRECTORIES ###
INDIR <- "datastore/raw/fannie_mae/data" # Directory containing the raw data files
OUTDIR <- "datastore/output/derived/fannie_mae" # Directory where output CSVs will be saved

# Create output directory if it doesn't exist
if (!dir.exists(OUTDIR)) {
    dir.create(OUTDIR, recursive = TRUE)
}

### Define the set of files to read in - Processing 2025Q1 only ###
starting_file <- 100 # 2025Q1 (year 25, quarter 1: 25*4 + 0 = 100)
ending_file <- 100

# Process the files (outputs to csv)
for (file_number in starting_file:ending_file) {
    # Set up file names
    fileYear <- file_number %/% 4
    if (nchar(fileYear) == 1) {
        fileYear <- paste0("0", fileYear)
    }
    fileYear <- paste0("20", fileYear)
    fileQtr <- (file_number %% 4) + 1
    fileQtr <- paste0("Q", fileQtr)
    FileName <- paste0(fileYear, fileQtr, ".csv")

    # Full path to input file
    full_input_path <- file.path(INDIR, FileName)

    # Check if file exists
    if (!file.exists(full_input_path)) {
        warning(paste("File not found:", full_input_path))
        next
    }

    cat("Processing:", FileName, "\n")

    # Run helper file (LPPUB_StatFile_Production.R must be in same directory as this script)
    source("source/derived/fannie_mae/LPPUB_StatFile_Production.R")
}
