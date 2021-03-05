library(earth)

# Load the Boston housing data
boston <- pdp::boston
boston$chas <- as.integer(boston$chas) - 1  # coerce to 0/1
X <- subset(boston, select = -cmedv)  # feature columns only

# Fit a third degree MARS model using 5-fold cross-validation
(boston.mars <- earth(cmedv ~ ., data = boston, degree = 3, pmethod = "cv",
                      nfold = 5, ncross = 10))

# Prediction wrapper for use with `shap$KernelExplainer()`
pfun.ks <- function(newdata) {  # Note: only a function of newdata!
  predict(mars, newdata = newdata)  
}  # returns an `nrow(newdata)` x 1 matrix of predictions

library(reticulate)

# Import {shap} and run KernelSHAP algorithm with 100 repetitions
shap <- import("shap")
explainer <- shap$KernelExplainer(pfun.ks, data = X)  # initialize explainer
system.time({  # time KernelSHAP
  ex.ks <- explainer$shap_values(X, nsamples = 100L)
})
ex.ks <- ex.ks[[1L]]  # results returned as a list
colnames(ex.ks) <- colnames(X)  # add column names
(ex.ks <- tibble::as_tibble(ex.ks))  # for nicer printing

# Save results
saveRDS(ex.ks, file = "data/ex_ks.rds")

# Run SampleSHAP algorithm with 100 repetitions
pfun.ss <- function(object, newdata) {
  predict(object, newdata = newdata)[, 1L, drop = TRUE]
}  # `fastshap::explain()` requires a vector of predictions
set.seed(1503)  # for reproducibility
system.time({  # time SampleSHAP
  ex.ss <- fastshap::explain(boston.mars, X = X, pred_wrapper = pfun.ss,
                             nsim = 100)
})
#   user  system elapsed 
# 14.055   0.581  14.628 

par(mfrow = c(1, 2))
plot(X$lstat, ex.ks[, "lstat"], col = 2)
plot(X$lstat, ex.ss$lstat)

cors <- sapply(1:ncol(ex2), FUN = function(i) cor(ex1[, i], ex2[, i]))
