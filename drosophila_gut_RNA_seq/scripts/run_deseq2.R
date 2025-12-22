#!/usr/bin/env Rscript

# Load required libraries
library(DESeq2)
library(tidyverse)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
counts_file <- args[1]
metadata_file <- args[2]
output_file <- args[3]

# Read count data and metadata
count_data <- read.table(counts_file, header = TRUE, row.names = 1, sep = "\t")
metadata <- read.table(metadata_file, header = TRUE, row.names = 1)

# Ensure sample names match between count data and metadata
count_data <- count_data[, rownames(metadata)]

# Create DESeq2 object
dds <- DESeqDataSetFromMatrix(
  countData = count_data,
  colData = metadata,
  design = ~ condition
)

# Filter low count genes
dds <- dds[rowSums(counts(dds) >= 10) >= 3, ]

# Run DESeq2
dds <- DESeq(dds)

# Get results
res <- results(dds, alpha = 0.05, lfcThreshold = 1)

# Convert to data frame and add gene names
res_df <- as.data.frame(res) %>%
  rownames_to_column("gene_id") %>%
  arrange(padj, pvalue)

# Save results
write.csv(res_df, file = output_file, row.names = FALSE)
