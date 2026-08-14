#!/usr/bin/env Rscript

library("data.table")
library("tidyverse")

l <- list.files(pattern = "*.hsq$")

# read_hsq <- function(f) {
#   fread(f, nrow = 10)[
#     Source == "V(G)/Vp",
#     .(
#       phenotype = str_remove(f, ".hsq$") |> str_remove("norm_"),
#       hsq = Variance,
#       se = SE
#     )
#   ]
# }

read_hsq <- function(f) {
  # dt <- fread(f, nrow = 10)
  dt <- read.table(f, header = TRUE, fill = TRUE)

  data.table(
    phenotype = str_remove(f, ".hsq$") |> str_remove("norm_"),
    hsq = dt[dt$Source == "V(G)/Vp", "Variance"],
    pval = dt[dt$Source == "Pval", "Variance"],
    se = dt[dt$Source == "V(G)/Vp", "SE"]
  )
}

df <- lapply(l, read_hsq) |> rbindlist()
df[, ci_95_low := hsq - qnorm(1 - 0.05 / 2) * se]
df[, ci_95_high := hsq + qnorm(1 - 0.05 / 2) * se]
# df[, hsq := round(hsq, 2)]
# df[, se := round(se, 2)]
df[, ci_95_low := round(ci_95_low, 2)]
df[, ci_95_high := round(ci_95_high, 2)]

fwrite(df[order(-df$hsq), ], "gcta_hsq_with_confint_pval.tsv", sep = "\t")

# filtered_phenotypes <- df %>%
#   filter(hsq > 0.11) %>%
#   arrange(desc(hsq)) %>%
#   select(phenotype) %>%
#   pull(phenotype)
# phenotypes_string <- paste(filtered_phenotypes, collapse = ", ")
