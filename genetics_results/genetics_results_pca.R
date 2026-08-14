### Compute PCA for all features, take top PC's explaining 95% of variance as phenotypes
### Note: PC's are not inverse rank normalized

library(factoextra)

setwd("/nfs/research/birney/users/esther/medaka-ziram/genetics_results/")

### Combine phenotypes, covariates to test normality
covar <- read.delim("input/raw/flexlmm_covariates_outlowcov_Gronske_F2.tsv", # 1835 samples
                    sep = "\t",
                    header = TRUE)

# pheno_name = "input/CNN_gcta.pheno_formatted"
pheno_name = "input/shapeembed_gcta.pheno_formatted"
# pheno_name = "input/vae_spring-sweep-14_gcta.pheno_formatted"
pheno <- read.delim(pheno_name, # 1658 samples
                    na.strings = c(""),
                    header = FALSE,
                    sep = "\t")
pheno <- pheno[complete.cases(pheno), ]
covar_pheno <- merge(covar, pheno, by.x="X.IID", by.y="V1") # 1460 samples
print(dim(covar_pheno))

# Compute PCA
covar_pheno.pheno <- covar_pheno[6:ncol(covar_pheno)]

pheno.prcomp <- prcomp(covar_pheno.pheno,
                       center = TRUE,
                       scale. = TRUE)

eig <- get_eig(pheno.prcomp)
# eig_bound <- which(eig$cumulative.variance.percent >= 95)[1]
eig_bound <- 20
fviz_eig(pheno.prcomp, addlabels = TRUE, choice = "variance", ncp = eig_bound,
         main = "")

# Save scree plot
results_dir <- sub(
  "^input/(.*)_gcta\\.pheno_formatted$",
  "hsq_\\1_pca_gcta",
  pheno_name
)

ggsave(paste0(results_dir, "/scree.png"), 
       width = 14, height = 6, dpi = 1200)

# Combine top 20 PC's and create new pheno df
covar_pheno_pca <- cbind(covar_pheno[1:4], pheno.prcomp$x[,1:20])
rownames(covar_pheno_pca) <- covar_pheno_pca$X.IID
covar_pheno_pca <- subset(covar_pheno_pca, select = -c(Cross, Folder, Tank))

write.table(covar_pheno_pca,
            paste0(pheno_name, "_pca"),
            sep = "\t",
            quote = FALSE,
            row.names = TRUE,
            col.names = FALSE)

### Plot hsq results
# Get hsq table
setwd("/nfs/research/birney/users/esther/medaka-ziram/genetics_results/")
# results_dir = "hsq_CNN_pca_gcta"
# results_dir = "hsq_shapeembed_pca_gcta"
results_dir = "hsq_vae_spring-sweep-14_pca_gcta"
hsq_results <- as.data.frame(read.delim(paste0(results_dir, "/gcta_hsq_with_confint_pval.tsv")))
hsq_results$phenotype_pc <- ifelse(
  grepl("^V[0-9]+$", hsq_results$phenotype),
  paste0("PC", as.integer(sub("^V", "",hsq_results$phenotype)) - 2),
  hsq_results$phenotype
)

# Filter for pval < 0.05
pval_thres <- 0.05#/nrow(hsq_results)
hsq_results <- hsq_results[hsq_results$pval < pval_thres, ]
print(dim(hsq_results))
print(dim(hsq_results[hsq_results$hsq >= 0.1208,])) # number of kinks baseline
print(dim(hsq_results[hsq_results$hsq >= 0.0942,])) # SC baseline

ggplot(hsq_results, aes(x = hsq, y = reorder(phenotype_pc, hsq), fill = pval < pval_thres/nrow(hsq_results))) +
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
    # axis.text.y = element_blank(),
    # axis.title = element_blank(),
    # axis.text = element_blank(),
    axis.ticks = element_blank(),
    legend.position = "none"
  )

ggsave(paste0(results_dir, "/hsq_pca.png"), 
       width = 6, height = 4, dpi = 1200)
