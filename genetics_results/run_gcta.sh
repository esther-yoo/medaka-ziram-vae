#!/bin/bash

for i in $(seq 0 19); do
  pheno=$i
  mpheno=$((i + 1))

  /hps/software/users/birney/esther/gcta/gcta64 --reml --pheno input/vae_spring-sweep-14_gcta.pheno_formatted_pca --covar input/gcta.cov --grm input/input --out hsq_vae_spring-sweep-14_pca_gcta/norm_V$((pheno + 3)) --mpheno $mpheno --reml-maxit 500
done
