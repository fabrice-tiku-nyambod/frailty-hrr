## 03_statistical_analysis.R
## RQ1: naive vs. motion-confirmed rest-onset timing (paired comparison + Bland-Altman)
## RQ2: block-wise linear mixed-effects models of delta_hr60_confirmed
## See manuscript_plan.md Section 6 and manuscript_methods_draft.md for the full
## rationale behind each modeling choice.

required_pkgs <- c("readr", "dplyr", "tidyr", "stringr", "ggplot2", "lme4", "lmerTest", "car")
missing_pkgs <- required_pkgs[!required_pkgs %in% installed.packages()[, "Package"]]
if (length(missing_pkgs) > 0) install.packages(missing_pkgs)

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(car)

dir.create("data_derived", showWarnings = FALSE)

## Mirror all console output (cat/print) to a log file as well as the
## console, so results can be read directly from disk instead of relying on
## copy-pasting console output (which has repeatedly hit truncation/stale-
## history issues with long R sessions). split=TRUE keeps it visible in
## RStudio too. on.exit() is unreliable at top-level script scope (it's
## meant for use inside functions), so sink() is closed explicitly at the
## very end of this script instead; the while-loop here defensively clears
## any sink left open by a previous run that errored out before reaching
## that point.
while (sink.number() > 0) sink()
log_con <- file("data_derived/rq_analysis_log.txt", open = "wt")
sink(log_con, split = TRUE)

## --- load ------------------------------------------------------------------

hrr <- read_csv("data_derived/hrr_features_all.csv",
                col_types = cols(patient_id = col_character(), .default = col_guess())) %>%
  mutate(patient_id = str_pad(patient_id, 3, pad = "0"))

subj <- readRDS("data_derived/subject_info_clean.rds") %>%
  mutate(patient_id = str_pad(patient_id, 3, pad = "0"))

## test-intensity ordering per manuscript_plan.md: TUG < STAIR < 6MWT < VELO
test_levels <- c("TUG", "STAIR", "6MWT", "VELO")
hrr <- hrr %>% mutate(test_type = factor(test_type, levels = test_levels))

## =============================================================================
## RQ1 (primary, methodological): naive vs. motion-confirmed timing/recovery
## =============================================================================

rq1 <- hrr %>% filter(!is.na(discrepancy_sec))
cat(sprintf("\nRQ1: n = %d test instances with both naive and confirmed timing computable\n", nrow(rq1)))

## timing discrepancy: is confirmed rest onset systematically later than naive?
wilcox_timing <- wilcox.test(rq1$discrepancy_sec, mu = 0, conf.int = TRUE)
cat("\n--- Wilcoxon signed-rank test: discrepancy_sec vs 0 ---\n")
print(wilcox_timing)
cat(sprintf("median discrepancy: %.1f s (IQR %.1f to %.1f)\n",
            median(rq1$discrepancy_sec),
            quantile(rq1$discrepancy_sec, 0.25), quantile(rq1$discrepancy_sec, 0.75)))

## paired comparison of recovery estimates (naive vs confirmed), where both exist
rq1_hr <- hrr %>% filter(!is.na(delta_hr60_naive), !is.na(delta_hr60_confirmed))
cat(sprintf("\nRQ1 (HRR paired comparison): n = %d\n", nrow(rq1_hr)))
wilcox_hrr <- wilcox.test(rq1_hr$delta_hr60_naive, rq1_hr$delta_hr60_confirmed, paired = TRUE, conf.int = TRUE)
cat("\n--- Wilcoxon signed-rank test: naive vs confirmed delta_hr60 (paired) ---\n")
print(wilcox_hrr)

## Bland-Altman: agreement between naive- and confirmed-timed HRR estimates
ba_data <- rq1_hr %>%
  mutate(mean_est = (delta_hr60_naive + delta_hr60_confirmed) / 2,
         diff_est = delta_hr60_naive - delta_hr60_confirmed)
ba_bias <- mean(ba_data$diff_est)
ba_sd <- sd(ba_data$diff_est)
ba_loa <- c(ba_bias - 1.96 * ba_sd, ba_bias + 1.96 * ba_sd)

p_bland_altman <- ggplot(ba_data, aes(x = mean_est, y = diff_est)) +
  geom_point(aes(color = test_type), alpha = 0.7) +
  geom_hline(yintercept = ba_bias, linetype = "solid", color = "black") +
  geom_hline(yintercept = ba_loa, linetype = "dashed", color = "gray40") +
  labs(x = "Mean of naive and confirmed ΔHR60 (bpm)",
       y = "Naive − confirmed ΔHR60 (bpm)",
       title = "Bland-Altman: naive vs. motion-confirmed HRR estimate",
       subtitle = sprintf("bias = %.2f bpm, 95%% limits of agreement [%.2f, %.2f]", ba_bias, ba_loa[1], ba_loa[2]),
       color = "Test type") +
  theme_minimal()
ggsave("data_derived/rq1_bland_altman.png", p_bland_altman, width = 7, height = 5, dpi = 600)
cat(sprintf("\nBland-Altman bias = %.2f bpm, limits of agreement [%.2f, %.2f]\n", ba_bias, ba_loa[1], ba_loa[2]))
cat("saved plot: data_derived/rq1_bland_altman.png\n")

## =============================================================================
## RQ2 (secondary, exploratory): block-wise mixed-effects models of
## delta_hr60_confirmed
## =============================================================================

## NYHA class and AF status are collapsed to binary groupings for modeling
## stability: NYHA I (n=3) and IV (n=1) are too sparse in this cohort to
## estimate as separate levels alongside an EFS x beta-blocker interaction
## (caused a rank-deficient design matrix / silently dropped coefficient in
## the 4-level version). NYHA I-II vs III-IV, and AF none vs any (permanent/
## persistent/paroxysmal combined), are standard, clinically-meaningful
## collapses, not arbitrary binning -- but this simplification should be
## stated explicitly in the manuscript, not left implicit.
model_data <- hrr %>%
  filter(!is.na(delta_hr60_confirmed)) %>%
  inner_join(subj, by = "patient_id") %>%
  mutate(
    test_duration_s = if_else(!is.na(t_still_sec), t_still_sec - onset_sec, NA_real_),
    beta_blockers = as.factor(beta_blockers),
    ## NB: `x %in% c(...)` treats NA as FALSE, not NA -- using it directly
    ## here would silently miscode missing nyha_class as "III-IV" instead of
    ## propagating NA, which is exactly what broke the earlier drop_na() (it
    ## never saw an NA to drop, so blocks with/without nyha_binary ended up
    ## fit on different row counts). case_when() keeps NA as NA explicitly.
    nyha_binary = case_when(
      is.na(nyha_class) ~ NA_character_,
      nyha_class %in% c("I", "II") ~ "I-II",
      TRUE ~ "III-IV"
    ),
    nyha_binary = factor(nyha_binary, levels = c("I-II", "III-IV")),
    af_binary = case_when(
      is.na(af_status) ~ NA_character_,
      af_status == "none" ~ "none",
      TRUE ~ "any"
    ),
    af_binary = factor(af_binary, levels = c("none", "any")),
    surgery_type = droplevels(surgery_type)
  ) %>%
  ## complete-case filter on every variable used in the FULL (Block D) model,
  ## applied before fitting ANY block, so all four models share identical
  ## rows and the likelihood-ratio comparison across blocks is valid
  drop_na(delta_hr60_confirmed, test_type, test_duration_s, efs_score,
          nyha_binary, af_binary, beta_blockers, age_years, days_after_surgery,
          surgery_type, copd, calcium_channel_blockers)

cat("\n--- factor level counts in final modeling dataset (check for sparse cells) ---\n")
print(table(model_data$nyha_binary, useNA = "always"))
print(table(model_data$af_binary, useNA = "always"))
print(table(model_data$beta_blockers, useNA = "always"))
print(table(model_data$surgery_type, useNA = "always"))

cat(sprintf("\nRQ2: modeling dataset n = %d test instances, %d patients\n",
            nrow(model_data), n_distinct(model_data$patient_id)))

## Block A: core (test type/intensity, test duration)
m_a <- lmer(delta_hr60_confirmed ~ test_type + test_duration_s + (1 | patient_id),
            data = model_data, REML = FALSE)

## Block B: + frailty
m_b <- lmer(delta_hr60_confirmed ~ test_type + test_duration_s + efs_score + (1 | patient_id),
            data = model_data, REML = FALSE)

## --- beta-blocker use: pre-specified per manuscript_plan.md Section 6, but
## excluded from the reported models below. 205 of 207 test instances (99%)
## were on beta-blockers -- consistent with standard post-cardiac-surgery
## care, but leaving essentially no contrast to estimate a main effect or an
## EFS interaction against. Attempting it anyway (kept here, not reported,
## for transparency/reproducibility) produces a rank-deficient design matrix
## and visibly destabilizes the rest of the model (e.g. the intercept SE
## roughly quadruples vs. the version without it). This is disclosed
## explicitly in the manuscript rather than silently omitted.
cat("\n--- beta-blocker prevalence (why it is excluded from reported models) ---\n")
print(table(model_data$beta_blockers, useNA = "always"))
m_c_attempt_with_beta_blockers <- lmer(
  delta_hr60_confirmed ~ test_type + test_duration_s + efs_score +
    nyha_binary + af_binary + beta_blockers * efs_score + age_years + days_after_surgery +
    (1 | patient_id),
  data = model_data, REML = FALSE)
cat("\n--- attempted model WITH beta_blockers (not used for reported results; see note above) ---\n")
print(summary(m_c_attempt_with_beta_blockers))

## Block C (reported): + pre-specified clinical correlates, beta-blocker term excluded
m_c <- lmer(delta_hr60_confirmed ~ test_type + test_duration_s + efs_score +
              nyha_binary + af_binary + age_years + days_after_surgery +
              (1 | patient_id),
            data = model_data, REML = FALSE)

## Block D (reported): + exploratory covariates, beta-blocker term excluded
m_d <- lmer(delta_hr60_confirmed ~ test_type + test_duration_s + efs_score +
              nyha_binary + af_binary + age_years + days_after_surgery +
              surgery_type + copd + calcium_channel_blockers +
              (1 | patient_id),
            data = model_data, REML = FALSE)

cat("\n--- rows used per model (must match for a valid LRT comparison) ---\n")
cat(sprintf("m_a: %d | m_b: %d | m_c: %d | m_d: %d | model_data: %d\n",
            nobs(m_a), nobs(m_b), nobs(m_c), nobs(m_d), nrow(model_data)))

cat("\n--- Model comparison (likelihood-ratio tests across blocks) ---\n")
print(anova(m_a, m_b, m_c, m_d))

cat("\n--- Block B summary (core + frailty; likely primary reported model) ---\n")
print(summary(m_b))

cat("\n--- Block C summary (+ clinical correlates) ---\n")
print(summary(m_c))

## diagnostics on the primary reported model (Block B)
cat("\n--- VIF, Block B fixed effects ---\n")
print(vif(m_b))

## base R's png() device defaults to pixels @72dpi with no res= set; specify
## units="in" + res=600 explicitly to get a true 600dpi print-quality image
## at the same physical size as the original (900x450px @72dpi = 12.5x6.25in)
png("data_derived/rq2_residual_diagnostics.png", width = 12.5, height = 6.25,
    units = "in", res = 600)
par(mfrow = c(1, 2))
plot(fitted(m_b), resid(m_b), main = "Residuals vs fitted (Block B)", xlab = "Fitted", ylab = "Residuals")
abline(h = 0, col = "red")
qqnorm(resid(m_b), main = "Normal Q-Q (Block B)")
qqline(resid(m_b), col = "red")
dev.off()
cat("saved plot: data_derived/rq2_residual_diagnostics.png\n")

## =============================================================================
## Fig 3 material: delta_hr60_confirmed by test type and by EFS category
## =============================================================================

model_data <- model_data %>%
  mutate(efs_tertile = ntile(efs_score, 3))

p_by_test <- ggplot(model_data, aes(x = test_type, y = delta_hr60_confirmed)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.15, alpha = 0.4) +
  labs(x = "Test type", y = "ΔHR60-confirmed (bpm)",
       title = "Residual excess HR by test type") +
  theme_minimal()

## RAW (unadjusted) view: pools all four test types together, which have
## very different baseline ΔHR60 magnitudes (TUG ~1 bpm vs 6MWT ~6-12 bpm)
## and differ somewhat in mean test duration across EFS tertiles -- both of
## which are exactly what Block A/B's fixed effects remove. Plotting the raw,
## pooled outcome against EFS tertile therefore visually understates (and in
## this cohort, nearly erases) the adjusted association, which is real within
## most individual test types (checked directly: 6MWT, STAIR, and TUG each
## show higher mean delta_hr60_confirmed in the top EFS tertile; VELO does
## not). Kept as a supplementary figure for transparency rather than dropped,
## since the raw/adjusted discrepancy itself is a finding worth showing, not
## hiding.
p_by_efs_raw <- ggplot(model_data, aes(x = factor(efs_tertile), y = delta_hr60_confirmed)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.15, alpha = 0.4) +
  labs(x = "EFS tertile (1 = least frail)", y = "ΔHR60-confirmed (bpm)",
       title = "Residual excess HR by frailty tertile (unadjusted)") +
  theme_minimal()

## ADJUSTED view: residualize out the Block A fixed effects (test type +
## test duration) using population-level (fixed-effects-only) predictions
## -- re.form=NA -- so between-patient variation (where the EFS signal
## lives) is retained rather than absorbed into the random intercept, which
## is what a plain resid() would do. This is the quantity that actually
## corresponds to what Table 3 / Block B reports.
model_data$fitted_fixed_only <- predict(m_a, re.form = NA)
model_data$resid_adjusted <- model_data$delta_hr60_confirmed - model_data$fitted_fixed_only

cat("\n--- adjusted residual (test type + duration removed) by EFS tertile ---\n")
print(model_data %>% group_by(efs_tertile) %>%
        summarise(mean_adj = mean(resid_adjusted), median_adj = median(resid_adjusted), n = n()))

## Fig 3b (effect plot, replaces the tertile boxplot): discretizing EFS into
## tertiles for plotting buried the very slope the model reports. This shows
## the continuous relationship instead -- individual adjusted residuals
## (points) plus a simple OLS line + 95% CI fit to those same points. The
## line is a visualization aid only, not the inferential model: it ignores
## the per-patient random intercept that the actual Block B mixed model
## accounts for, and its slope will not exactly equal the +0.64 bpm/point
## Table 3 estimate, though it is expected to be close and same-signed. This
## is disclosed in the figure caption. Showing the points (not just the
## line) is deliberate -- it lets the reader see directly that the upward
## trend is carried by a subset of highly-frail patients with large
## responses, not a uniform shift (Section 3.3), which the tertile boxplot
## obscured in the opposite direction (making the trend look absent).
p_by_efs_adj <- ggplot(model_data, aes(x = efs_score, y = resid_adjusted)) +
  geom_point(alpha = 0.45, size = 1.8) +
  geom_smooth(method = "lm", se = TRUE, color = "#2a78d6", fill = "#2a78d6", alpha = 0.18) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(x = "Edmonton Frail Scale score",
       y = "ΔHR60-confirmed, adjusted for\ntest type & duration (bpm)",
       title = "Adjusted residual excess HR vs. frailty severity") +
  theme_minimal(base_size = 12)

ggsave("data_derived/fig3a_delta_by_test_type.png", p_by_test, width = 6, height = 4.5, dpi = 600)
ggsave("data_derived/fig3b_delta_by_efs_tertile_adjusted.png", p_by_efs_adj, width = 6, height = 4.5, dpi = 600)
ggsave("data_derived/figS3_delta_by_efs_tertile_raw.png", p_by_efs_raw, width = 6, height = 4.5, dpi = 600)
cat("saved plots: fig3a (test type), fig3b (adjusted EFS, continuous effect plot), figS3 (raw/unadjusted EFS, supplementary)\n")

cat("\nDone.\n")

sink()
close(log_con)
