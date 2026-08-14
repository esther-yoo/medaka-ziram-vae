#!/usr/bin/env Rscript
library("data.table")
setwd("/nfs/research/birney/users/esther/medaka-ziram/genetics_results/")

### Make lower-triangular GRM from full GRM
# samples <- fread("./input.rel.id")[["#IID"]]
# K <- matrix(
#   readBin("./input.rel.bin", what = "numeric", n = length(samples)**2),
#   ncol = length(samples)
# )
# stopifnot(sum(is.na(K)) == 0)
# 
# colnames(K) <- samples
# rownames(K) <- samples
# 
# n <- nrow(K)
# k_vec_list <- lapply(
#   1:n,
#   function(i) {
#     sapply(
#       1:i,
#       function(j, i) {
#         K[i, j]
#       },
#       i = i
#     )
#   }
# )
# k_vec <- unlist(k_vec_list)
# handle <- file("input.grm.bin", "wb")
# writeBin(k_vec, handle, size = 4)
# close(handle)
# 
# fwrite(
#   list(colnames(K), colnames(K)),
#   "input.grm.id",
#   sep = "\t",
#   col.names = FALSE
# )


### Format covariates file for GCTA
covar <- read.delim("input/raw/flexlmm_covariates_outlowcov_Gronske_F2.tsv",
                    sep = "\t",
                    header = TRUE)
covar$X.IID2 <- covar$X.IID
covar <- covar[, c("X.IID", "X.IID2", "Tank", "Folder")] # Add back Tank, Folder for test set
# covar <- covar[, c("X.IID", "X.IID2", "Cross", "Tank", "Folder")]
write.table(covar,
            "input/gcta.cov",
            sep = "\t",
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE)


##### Format phenotypes for GCTA
img_to_name_map <- read.delim("input/raw/Gronske_IID_imgpth_full_F2_merged.csv",
                              sep = ",")
img_to_name_map$merged <- NULL
# img_to_name_map$img_path_basename <- sub(
#   "^([^\\-]+)---(.*)$",
#   "-\\2--\\1",
#   sub("\\.tif$", "", basename(img_to_name_map$image_path))
# )
img_to_name_map$img_path_basename <- sub("\\.tif$", "", basename(img_to_name_map$image_path))

### RegionProps
# pheno_name = "RegionProps_gcta.pheno"
# pheno <- read.delim("input/raw/RegionProps_baseline_HOLDOUT_test.csv",
#                     sep = ",")
# pheno$mask <- NULL
# pheno$CO6 <- NULL
# pheno_out <- merge(img_to_name_map,
#                    pheno,
#                    by.x = "img_path_basename",
#                    by.y = "stem")
# pheno_out <- pheno_out[complete.cases(pheno_out), ]
# pheno_out <- pheno_out[pheno_out$X.IID != "", ] # 1658 samples remaining
# rownames(pheno_out) <- pheno_out$X.IID
# pheno_out$img_path_basename <- NULL
# pheno_out$image_path <- NULL

### CNN
# pheno_name = "CNN_gcta.pheno"
# pheno <- read.delim("input/raw/CNN_features_HOLDOUT_Test.csv",
#                     sep = ",")
# pheno$label <- NULL # Remove label
# pheno_out <- merge(img_to_name_map,
#                    pheno,
#                    by.x = "img_path_basename",
#                    by.y = "fish_id")#
# pheno_out <- pheno_out[complete.cases(pheno_out), ]
# pheno_out <- pheno_out[pheno_out$X.IID != "", ] # 1658 samples remaining
# rownames(pheno_out) <- pheno_out$X.IID
# pheno_out$img_path_basename <- NULL
# pheno_out$image_path <- NULL

### ShapeEmbed
# pheno_name = "shapeembed_gcta.pheno"
# pheno <- read.delim("input/raw/ShapeEmbed_features_HOLDOUT_Test.csv",
#                     sep = ",")
# pheno$label <- NULL # Remove label
# pheno_out <- merge(img_to_name_map,
#                    pheno,
#                    by.x = "img_path_basename",
#                    by.y = "fish_id")
# pheno_out <- pheno_out[complete.cases(pheno_out), ]
# pheno_out <- pheno_out[pheno_out$X.IID != "", ] # 1658 samples remaining
# rownames(pheno_out) <- pheno_out$X.IID
# pheno_out$img_path_basename <- NULL
# pheno_out$image_path <- NULL

### VAE
vae_run_name = "spring-sweep-14" # spring-sweep-14
pheno_name = paste0("vae_", vae_run_name, "_gcta.pheno")
pheno <- read.delim(paste0("input/raw/VAE_", vae_run_name, "_features_HOLDOUT_Test.csv"),
                    sep = ",")
pheno$severity_score_adjusted <- NULL # Remove label
pheno_out <- merge(img_to_name_map,
                   pheno,
                   by.x = "img_path_basename",
                   by.y = "img_path")
pheno_out <- pheno_out[complete.cases(pheno_out), ]
pheno_out <- pheno_out[pheno_out$X.IID != "", ] # 1658 samples remaining for both
rownames(pheno_out) <- pheno_out$X.IID
pheno_out$img_path_basename <- NULL
pheno_out$image_path <- NULL

write.table(pheno_out,
            paste0("input/", pheno_name, "_formatted"),
            sep = "\t",
            quote = FALSE,
            row.names = TRUE,
            col.names = FALSE)

### Read back in phenotypes to correspond them to hsq phenotype number
# pheno_name = "input/raw/shapeEmbed_features_val.csv"
# pheno <- read.delim(pheno_name,
#                     na.strings = c(""),
#                     sep=",")
# pheno <- pheno[complete.cases(pheno), ]
# rownames(pheno) <- pheno$X.IID


pheno_out$X.IID <- NULL
old_colnames <- colnames(pheno_out)
new_colnames <- paste0("V", seq_len(ncol(pheno_out))+2) # For GCTA input files, first two columns are the sample name
colnames_map <- cbind(old_colnames, new_colnames)
write.table(colnames_map,
            paste0("hsq_", gsub("\\.pheno$", "", basename(pheno_name)), "/colnames_map"),
            row.names = FALSE,
            col.names = FALSE,
            sep = "\t",
            quote = FALSE)

### Test normality of residuals after regressing fixed covariates
### Inverse rank normalization function:
inv_norm <- function(x) {
  ranks <- rank(x, ties.method = "average")
  percentiles <- (ranks - 0.5) / length(x)
  ret <- qnorm(percentiles)
  return(ret)
}

### Combine phenotypes, covariates to test normality
covar <- read.delim("input/raw/flexlmm_covariates_outlowcov_Gronske_F2.tsv",
                    sep = "\t",
                    header = TRUE)
pheno_name = "input/kinks_gcta.pheno_formatted"
pheno <- read.delim(pheno_name,
                    na.strings = c(""),
                    header = FALSE,
                    sep = "\t")
pheno <- pheno[complete.cases(pheno), ]
covar_pheno <- merge(covar, pheno, by.x="X.IID", by.y="V1")

covar_pheno_norm <- covar_pheno # To store the rank-normalized version
for (i in 3:((ncol(covar_pheno)-3))){
  print(i)
  formula <- as.formula(
    # paste0("V", i, " ~ Cross + Folder + Tank")
    paste0("V", i, " ~ Tank + Folder")
  )

  model <- lm(formula, data = covar_pheno)
  covar_pheno$e <- resid(model)
  
  norm_test <- shapiro.test(covar_pheno$e)
  if (norm_test$p.value < 0.05){
    print(paste0("V", i))
    print(norm_test)
    
    covar_pheno_norm[, paste0("V", i)] <- inv_norm(covar_pheno_norm[, paste0("V", i)])
  }

  # print(shapiro.test(covar_pheno$e))
  # print(mean(covar_pheno$e))
  # hist(covar_pheno$e)
}

# Test again after rank normalization
for (i in 3:((ncol(covar_pheno_norm)-3))){
  formula <- as.formula(
    # paste0("V", i, " ~ Cross + Folder + Tank")
    paste0("V", i, " ~ Tank + Folder")
  )
  
  model <- lm(formula, data = covar_pheno_norm)
  # model <- lm("f195 ~ Cross + Folder + Tank", data = covar_pheno)
  covar_pheno_norm$e <- resid(model)
  
  norm_test <- shapiro.test(covar_pheno_norm$e)
  if (norm_test$p.value < 0.05){
    print(paste0("V", i))
    print(norm_test)
  }
  
  # print(shapiro.test(covar_pheno$e))
  # print(mean(covar_pheno$e))
  # hist(covar_pheno$e)
}

# Save new inverse normalized df
pheno_norm <- subset(covar_pheno_norm, select = -c(Cross, Folder, Tank, e))

write.table(pheno_norm,
            paste0(pheno_name, "_norm"),
            sep = "\t",
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE)

#########################

# Plot Q-Q
library(ggplot2)

ggplot(covar_pheno_norm, aes(sample = e)) +
  stat_qq(color = "steelblue") +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(
    title = "Q-Q Plot",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  ) +
  theme_minimal()


### Plot barplot of hsq results
library(ggplot2)
library(reshape2)

# Get hsq table
setwd("/nfs/research/birney/users/esther/medaka-ziram/genetics_results/")

### all hsq
# results_dir = "hsq_RegionProps_gcta"
# results_dir = "hsq_CNN_gcta"
# results_dir = "hsq_shapeembed_gcta"
results_dir = "hsq_vae_spring-sweep-14_gcta"

### shape residuals
# results_dir = "hsq_CNN_shaperesid_gcta"
# results_dir = "hsq_shapeembed_shaperesid_gcta"
# results_dir = "hsq_vae_spring-sweep-14_shaperesid_gcta"


hsq_results <- as.data.frame(read.delim(paste0(results_dir, "/gcta_hsq_with_confint_pval.tsv")))

# Filter for pval < 0.05
pval_thres <- 0.05#/nrow(hsq_results)
hsq_results <- hsq_results[hsq_results$pval < pval_thres, ]
print(dim(hsq_results))
print(dim(hsq_results[hsq_results$hsq >= 0.1208,])) # number of kinks baseline
print(dim(hsq_results[hsq_results$hsq >= 0.0942,])) # SC baseline
print(max(hsq_results[hsq_results$hsq >= 0.0942, "hsq"]))


# If using regionprops features, map feature numbers back to labels
if (results_dir == "hsq_RegionProps_gcta"){
  colnames_map = read.delim(paste0(results_dir, "/colnames_map"), header = FALSE)
  hsq_results <- merge(hsq_results,
                       colnames_map,
                       by.x = "phenotype",
                       by.y = "V2")
  hsq_results$phenotype <- NULL
  names(hsq_results)[names(hsq_results) == "V1"] <- "phenotype"
}

# Remove highly correlated (> 0.95) features:
### regionprops
# hsq_results <- hsq_results[!(hsq_results$phenotype %in% c("axis_major_length",
#                                                           "hu_moment_5",
#                                                           "hu_moment_2")), ]
### shape embed
# hsq_results <- hsq_results[!(hsq_results$phenotype %in% c("V18", "V121", "V7","V92","V69","V100","V29","V20","V43")), ]


# Plot hsq barplot
ggplot(hsq_results, aes(x = hsq, y = reorder(phenotype, hsq), fill = pval < pval_thres/nrow(hsq_results))) +
  # geom_col(fill = "steelblue") +
  geom_col() +
  scale_fill_manual(values = c("FALSE" = "grey70", "TRUE" = "steelblue")) +
  labs(
    # title = paste0(results_dir, " (test set F2's)"),
    x = expression(h^2),
    y = "Feature dimension"
  ) +
  geom_vline(
    xintercept = 0.0942,
    colour = "black",
    linewidth = 0.3,
    linetype = "dashed"
  ) +
  geom_vline(
    xintercept = 0.1208,
    colour = "black",
    linewidth = 0.3,
    linetype = "dashed"
  ) +
  # annotate(
  #   "segment",
  #   x = 0.0942, xend = 0.0942,
  #   y = 0.3, yend = 1,
  #   colour = "red",
  #   linewidth = 1,
  #   arrow = arrow(length = unit(0.2, "cm"))
  # ) +
  coord_cartesian(xlim = c(0, 0.2)) +
  theme_minimal() +
  theme(
    # axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    # axis.title = element_blank(),
    # axis.text = element_blank(),
    axis.ticks = element_blank(),
    legend.position = "none"
  )


ggsave(paste0(results_dir, "/hsq_color_pvalnorm.png"), 
       width = 6, height = 4, dpi = 1200)


#########################

# Plot latent space correlation matrix
# pheno_name = "input/vae_spring-sweep-14_gcta.pheno_formatted_shaperesid_norm"
pheno_name = "input/RegionProps_gcta.pheno_formatted_norm"
pheno <- read.delim(pheno_name,
                    na.strings = c(""),
                    header = FALSE,
                    sep = "\t")
pheno <- pheno[complete.cases(pheno), ]

library(corrplot)
# library(dplyr)
latent_df <- pheno %>% select(where(is.numeric))

# Map pheno colnames to original (only needed for regionprops)
latent_df_colnames <- read.delim(
  paste0(
    "hsq_RegionProps_gcta/colnames_map"),
  header = FALSE,
  col.names = c("orig", "new")
)
lookup <- setNames(latent_df_colnames$orig, latent_df_colnames$new)

names(latent_df) <- ifelse(
  names(latent_df) %in% names(lookup),
  lookup[names(latent_df)],
  names(latent_df)
)

# Reorder hsq results with highest hsq first
hsq_results <- hsq_results[order(hsq_results$hsq, decreasing = TRUE), ]

# Reorder latent_df columns
latent_df <- latent_df[, hsq_results$phenotype]

corr_mat <- cor(latent_df, use = "pairwise.complete.obs")

png(paste0(results_dir,"/correlation_matrix.png"), width = 2400, height = 2000, res = 1200)
corrplot(
  corr_mat,
  method = "color",
  # type = "full",
  order = "original",
  tl.pos = "n",
  # tl.pos = "lt",    # labels on left and top
  # tl.cex = 0.2,     # label size
  # tl.col = "black",  # label color
  cl.cex = 0.3,
  col = colorRampPalette(c("blue", "white", "red"))(200)
)
dev.off()

threshold <- 0.90

# Indices of upper triangle (excluding diagonal) above threshold
idx <- which(
  abs(corr_mat) > threshold & upper.tri(corr_mat),
  arr.ind = TRUE
)

# Convert to a data frame
high_corr_pairs <- data.frame(
  feature1 = rownames(corr_mat)[idx[, 1]],
  feature2 = colnames(corr_mat)[idx[, 2]],
  correlation = corr_mat[idx]
)

high_corr_pairs

### Get list of phenotypes to drop (highly correlated to higher h^2 phenotypes)
library(igraph)

# Make graph of correlated phenotypes
g <- graph_from_data_frame(
  high_corr_pairs[, c("feature1", "feature2")],
  directed = FALSE
)

# Find connected components (correlated groups)
groups <- components(g)$membership

# Phenotypes in each group
clusters <- split(names(groups), groups)

# Keep first phenotype in each cluster (because original df was sorted)
keep <- sapply(clusters, `[`, 1)

# Drop all others
drop <- setdiff(unlist(clusters), keep)

drop
