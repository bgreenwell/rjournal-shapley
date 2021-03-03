library(fastshap)  # for explain()
library(ggplot2)   # for autoplot()
library(ranger)    # for efficent random forest algorithm


# Set ggplot2 theme
theme_set(theme_bw())

# Load Ames housing data
ames <- as.data.frame(AmesHousing::make_ames())

# Create data frame of only features
X <- subset(ames, select = -Sale_Price)

# Fit a (default) random forest
set.seed(1644)  # for reproducibility
(rfo <- ranger(Sale_Price ~ ., data = ames))

# Prediction wrapper
pfun <- function(object, newdata) {
  predict(object, data = newdata)$predictions
}

# Grab training predictions
pred <- pfun(rfo, newdata = X)
extremes <- c(which.min(pred), which.max(pred))

# Check explanations
X[extremes[1L], ]$Overall_Qual
X[extremes[2L], ]$Overall_Qual

# Feature contributions for extreme predictions
ex.min <- explain(rfo, X = X, newdata = X[extremes[1L], ], nsim = 100,
                  pred_wrapper = pfun)
ex.max <- explain(rfo, X = X, newdata = X[extremes[2L], ], nsim = 100,
                  pred_wrapper = pfun)
p.min <- autoplot(ex.min, type = "contribution")
p.max <- autoplot(ex.max, type = "contribution")
gridExtra::grid.arrange(p.min, p.max, nrow = 1)

# Explain entire data set (useful for aggregated model summaries)
library(doParallel)
nc <- 8  # number of cores to use
cl <- if (.Platform$OS.type == "unix") nc else makeCluster(nc)
registerDoParallel(cl)
system.time({  # time expression
  ex.all <- explain(rfo, X = X, nsim = 100, pred_wrapper = pfun, adjust = TRUE,
                    .parallel = TRUE)
})
# >  nc
# [1] 8
#      user    system   elapsed 
# 12347.025   152.371  3386.896 

# Save results
saveRDS(ex.all, file = "ex-ames.rds")

# Shapley summary plot
p1 <- autoplot(ex.all, num_features = 20)
p2 <- autoplot(ex.all, type = "dependence", feature = "Gr_Liv_Area", X = X,
               color_by = "Exter_Qual", alpha = 0.3) +
  theme(legend.position = c(0.7, 0.2))
gridExtra::grid.arrange(p1, p2, nrow = 1)
