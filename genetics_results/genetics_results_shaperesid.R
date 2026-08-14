### Regress out regionprops features out of all features, and prepare the residuals as phenotypes
setwd("/nfs/research/birney/users/esther/medaka-ziram/genetics_results/")
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
regionprops <- read.delim("input/RegionProps_gcta.pheno_formatted",
                          na.strings = c(""),
                          header = FALSE,
                          sep = "\t")
latent_df_colnames <- read.delim(
  "/nfs/research/birney/users/esther/medaka-ziram/genetics_results/hsq_RegionProps_gcta/colnames_map",
  header = FALSE,
  col.names = c("orig", "new")
)
lookup <- setNames(latent_df_colnames$orig, latent_df_colnames$new)

names(regionprops) <- ifelse(
  names(regionprops) %in% names(lookup),
  lookup[names(regionprops)],
  names(regionprops)
)


pheno_name = "input/vae_spring-sweep-14_gcta.pheno_formatted"
pheno <- read.delim(pheno_name,
                    na.strings = c(""),
                    header = FALSE,
                    sep = "\t")
pheno <- pheno[complete.cases(pheno), ]
pheno_regionprops <- merge(regionprops, pheno, by = "V1")
pheno_regionprops$V2.x <- NULL
pheno_regionprops$V2.y <- NULL


# For each feature, regress out the regionprops features
for (i in 3:ncol(pheno)){
  print(i)
  formula <- as.formula(
    paste0("V", i, "~ area + convex_area + perimeter + axis_major_length + axis_minor_length + extent + eccentricity + 
                      solidity + feret_diameter_max + hu_moment_0 + hu_moment_1 + hu_moment_2 + hu_moment_3 + hu_moment_4 +
                      hu_moment_5 + hu_moment_6 + bbox_width + bbox_height + bbox_aspect_ratio")
  )
  
  model <- lm(formula, data = pheno_regionprops)
  pheno_regionprops[,paste0("V", i)] <- resid(model)
}

pheno_regionprops <- pheno_regionprops[, !(names(pheno_regionprops) %in% c("area", "convex_area", "perimeter", "axis_major_length", 
                                                                           "axis_minor_length", "extent", "eccentricity", "solidity",
                                                                           "feret_diameter_max", "hu_moment_0", "hu_moment_1", "hu_moment_2",
                                                                           "hu_moment_3", "hu_moment_4", "hu_moment_5", "hu_moment_6", 
                                                                           "bbox_width", "bbox_height", "bbox_aspect_ratio"))]
covar_pheno <- merge(covar, pheno_regionprops, by.x="X.IID", by.y="V1")

covar_pheno_norm <- covar_pheno # To store the rank-normalized version
for (i in 3:((ncol(pheno)))){
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
rownames(pheno_norm) <- pheno_norm$X.IID

write.table(pheno_norm,
            paste0(pheno_name, "_shaperesid_norm"),
            sep = "\t",
            quote = FALSE,
            row.names = TRUE,
            col.names = FALSE)
