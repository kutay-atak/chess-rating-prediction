# Latent skill model: does skill split into Accuracy and TimeUse, and does that hold across time controls?
# Reads sem_data.csv (sem_data.py); writes results/sem_fit.csv and results/sem_invariance.csv.
# Usage: Rscript skill_sem.R
library(lavaan)

d <- read.csv("sem_data.csv")
vars <- setdiff(names(d), "tc")
# z-scores on the pooled data: the raw variances span a factor of 6000 and stall the optimizer; one linear
# rescaling shared by every group changes neither the fit nor the invariance tests
d[vars] <- scale(d[vars])

measurement <- "
  Accuracy =~ log_avg_loss + mistake_rate + blunder_rate
  TimeUse  =~ mean_log_spent + sd_log_spent + low_clock_share
  log_avg_loss ~~ 0*log_avg_loss
  mean_log_spent ~~ 0*mean_log_spent
"
model <- paste(measurement, "rating ~ Accuracy + TimeUse")

# The two residual variances fixed at 0 above: left free, both come out negative (Heywood cases: log_avg_loss is
# nearly mistakes + blunders, and the time indicators barely correlate); bounding them at 0 lands exactly there,
# and fixing them (unlike bounds) also works with the equality constraints of the invariance models.
# MLR: the rates are skewed, so robust standard errors and a scaled chi-square
fit <- sem(model, data = d, estimator = "MLR")

measures <- c("chisq.scaled", "df.scaled", "pvalue.scaled", "cfi.robust", "tli.robust", "rmsea.robust", "srmr")
fm <- fitMeasures(fit, measures)
std <- standardizedSolution(fit)
keep <- std$op %in% c("=~", "~") | (std$op == "~~" & std$lhs == "Accuracy" & std$rhs == "TimeUse")
paths <- std[keep, ]
out <- rbind(
  data.frame(name = "n", value = nobs(fit)),
  data.frame(name = names(fm), value = as.numeric(fm)),
  data.frame(name = paste(paths$lhs, paths$op, paths$rhs), value = paths$est.std),
  data.frame(name = "R2 rating", value = inspect(fit, "r2")[["rating"]])
)
dir.create("results", showWarnings = FALSE)
write.csv(out, "results/sem_fit.csv", row.names = FALSE)
print(out, digits = 3)

# Measurement invariance of the two factors across the three most common time controls
top <- names(sort(table(d$tc), decreasing = TRUE))[1:3]
g <- d[d$tc %in% top, ]
steps <- list(configural = character(0), metric = "loadings", scalar = c("loadings", "intercepts"))
inv <- do.call(rbind, lapply(names(steps), function(s) {
  f <- cfa(measurement, data = g, group = "tc", group.equal = steps[[s]], estimator = "MLR")
  m <- fitMeasures(f, c("chisq.scaled", "df.scaled", "cfi.robust", "rmsea.robust", "srmr"))
  data.frame(model = s, as.list(m))
}))
# a CFI drop of at most 0.01 keeps the stricter model (Cheung & Rensvold 2002);
# with n in the hundreds of thousands every chi-square difference is significant
inv$delta_cfi <- c(NA, diff(inv$cfi.robust))
inv$groups <- paste(top, collapse = " / ")
write.csv(inv, "results/sem_invariance.csv", row.names = FALSE)
print(inv, digits = 3)
