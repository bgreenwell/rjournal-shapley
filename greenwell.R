## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  cache = TRUE,
  fig.width = 6,
  fig.asp = 0.618,
  out.width = "80%",
  fig.align = "center",
  fig.pos = "!htb",
  message = FALSE,
  warning = FALSE
)


## ----titanic-load-------------------------------------------------------------
# Read in the data and clean it up a bit
titanic <- titanic::titanic_train
features <- c(
  "Survived",  # passenger survival indicator
  "Pclass",    # passenger class
  "Sex",       # gender
  "Age",       # age
  "SibSp",     # number of siblings/spouses aboard
  "Parch",     # number of parents/children aboard
  "Fare",      # passenger fare
  "Embarked"   # port of embarkation
)
titanic <- titanic[, features]
titanic$Survived <- as.factor(titanic$Survived)
titanic <- na.omit(titanic)

# Data frame containing just the features
X <- subset(titanic, select = -Survived)


## ----titanic-glm--------------------------------------------------------------
fit <- glm(Survived ~ ., data = titanic, family = binomial)


## ----titanic-jack-------------------------------------------------------------
jack <- data.frame(
  Pclass = 3,
  Sex = factor("male", levels = c("female", "male")),
  Age = 20,
  SibSp = 0,
  Parch = 0,
  Fare = 15,  # lower end of third-class ticket prices
  Embarked = factor("S", levels = c("", "C", "Q", "S"))
)


## ----titanic-jack-predict-----------------------------------------------------
predict(fit, newdata = jack)


## ----titanic-helpers----------------------------------------------------------
# Prediction wrapper to compute predicted probability of survive
pfun <- function(object, newdata) {
  predict(object, newdata = newdata)
}

# DALEX-based helper for iBreakDown
explainer <- DALEX::explain(fit, data = X, y = titanic$Survived,                                             predict_function = pfun, verbose = FALSE)

# Helper for iml
predictor <- iml::Predictor$new(fit, data = titanic, y = "Survived",
                                predict.fun = pfun)


## ----titanic-jack-explanations, fig.width=9, fig.height=3, out.width="100%", fig.cap="TBD."----
# Compute explanations
set.seed(1039)  # for reproducibility
ex1 <- iBreakDown::shap(explainer, B = 100, new_observation = jack)
ex2 <- iml::Shapley$new(predictor, x.interest = jack, sample.size = 100)
ex3 <- fastshap::explain(fit, X = X, pred_wrapper = pfun, nsim = 100,
                         newdata = jack)

# Plot results
library(ggplot2)  # for `autoplot()` function
p3 <- plot(ex1) + ggtitle("iBreakDown")
p2 <- plot(ex2) + ggtitle("iml")
p1 <- autoplot(ex3, type = "contribution") + ggtitle("fastshap")
fastshap::grid.arrange(p1, p2, p3, nrow = 1)


## ----titanic-benchmark, fig.cap="Quick benchmark between three different implementations of SampleSHAP for explaining Jack's unfortunate prediction."----
nsims <- c(1, 5, 10, 25, 50, 75, seq(from = 100, to = 1000, by = 100))
times1 <- times2 <- times3 <- numeric(length(nsims))
set.seed(904)
for (i in seq_along(nsims)) {
  message("nsim = ", nsims[i], "...")
  times1[i] <- system.time({
    iBreakDown::shap(explainer, B = nsims[i], new_observation = jack)
  })["elapsed"]
  times2[i] <- system.time({
    iml::Shapley$new(predictor, x.interest = jack, sample.size = nsims[i])
  })["elapsed"]
  times3[i] <- system.time({
    fastshap::explain(fit, X = X, newdata = jack, pred_wrapper = pfun, 
                      nsim = nsims[i])
  })["elapsed"]
}
pal <- palette.colors(3, palette = "Okabe-Ito")  # colorblind friendly palette 
plot(nsims, times1, type = "b", xlab = "Number of Monte Carlo repetitions",
     ylab = "Time (in seconds)", las = 1, pch = 19, col = pal[1L],
     xlim = c(0, max(nsims)), ylim = c(0, max(times1, times2, times3)))
lines(nsims, times2, type = "b", pch = 19, col = pal[2L],)
lines(nsims, times3, type = "b", pch = 19, col = pal[3L],)
legend("topleft",
       legend = c("iBreakDown", "iml", "fastshap"),
       lty = 1, pch = 19, col = pal, inset = 0.02)


## ----titanic-exact------------------------------------------------------------
fastshap::explain(fit, newdata = jack, exact = TRUE)  # ExactSHAP
fastshap::explain(fit, X = X, pred_wrapper = pfun, nsim = 10000,
                  newdata = jack)  # SampleSHAP
predict(fit, newdata = jack, type = "terms")  # ExactSHAP (base R)

