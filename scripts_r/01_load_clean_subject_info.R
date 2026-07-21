## 01_load_clean_subject_info.R
## Loads subject-info.csv (two-row header, mean±std strings, decimal commas,
## "-" missingness) into a clean, analysis-ready tibble.

library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)

raw_path <- "extracted/wearable-based-signals-during-physical-exercises-from-patients-with-frailty-after-open-heart-surgery-1.0.0/subject-info.csv"

clean_names <- c(
  "patient_id", "age_years", "gender", "height_cm", "weight_kg",
  "efs_score", "days_after_surgery", "surgery_type", "nyha_class",
  "af_status", "copd", "depression", "musculoskeletal_disease",
  "oncological_disease", "ace_inhibitors", "beta_blockers",
  "calcium_channel_blockers", "mwt6_distance_m", "tug_time_s",
  "velo_duration_mmss", "velo_max_load_watt", "velo_max_hr_bpm",
  "step_length_left_cm", "step_length_right_cm", "stride_length_cm",
  "step_width_cm", "stance_phase_left_pct", "stance_phase_right_pct",
  "swing_phase_left_pct", "swing_phase_right_pct", "double_stance_phase_pct",
  "step_time_left_s", "step_time_right_s", "stride_time_s",
  "cadence_steps_min", "velocity_kmh", "gait_line_len_left_mm",
  "gait_line_len_right_mm", "single_limb_support_left_mm",
  "single_limb_support_right_mm", "ant_post_position_mm",
  "lateral_symmetry_mm", "max_gait_line_velocity_cms"
)

## Columns 23:43 are reported as "mean±std" (occasionally a bare number with
## no ± part, e.g. patient 254). These get split into _mean/_sd pairs.
mean_sd_cols <- clean_names[23:43]

## --- read, skipping BOTH header rows (group-label row + field-name row),
## since we supply col_names ourselves and don't want either header row
## mistaken for a data row.
raw <- read_csv(raw_path, skip = 2, col_names = clean_names,
                 col_types = cols(.default = col_character()),
                 na = c("-", ""), show_col_types = FALSE)

## European decimal commas (e.g. patient 203 weight "84,5") appear inconsistently;
## safe to normalize everywhere since no column uses comma as a thousands separator.
raw <- raw %>% mutate(across(everything(), ~ str_replace_all(.x, ",", ".")))

split_mean_sd <- function(x) {
  m <- str_match(x, "^(-?[0-9]*\\.?[0-9]+)(?:±(-?[0-9]*\\.?[0-9]+))?$")
  tibble(mean = as.numeric(m[, 2]), sd = as.numeric(m[, 3]))
}

mmss_to_sec <- function(x) {
  parts <- str_split(x, ":")
  map_dbl(parts, function(p) {
    if (length(p) != 2 || anyNA(p)) return(NA_real_)
    as.numeric(p[1]) * 60 + as.numeric(p[2])
  })
}

subject_info <- raw %>%
  mutate(
    age_years = as.numeric(age_years),
    gender = factor(gender, levels = c("0", "1"), labels = c("male", "female")),
    height_cm = as.numeric(height_cm),
    weight_kg = as.numeric(weight_kg),
    efs_score = as.numeric(efs_score),
    days_after_surgery = as.numeric(days_after_surgery),
    surgery_type = factor(surgery_type, levels = c("0", "1", "2"),
                          labels = c("CABG", "isolated_valve", "combined")),
    nyha_class = factor(nyha_class, levels = c("I", "II", "III", "IV"), ordered = TRUE),
    af_status = factor(af_status, levels = c("0", "1", "2", "3"),
                       labels = c("none", "permanent", "persistent", "paroxysmal")),
    across(c(copd, depression, musculoskeletal_disease, oncological_disease,
             ace_inhibitors, beta_blockers, calcium_channel_blockers),
           ~ as.logical(as.integer(.x))),
    mwt6_distance_m = as.numeric(mwt6_distance_m),
    tug_time_s = as.numeric(tug_time_s),
    velo_duration_s = mmss_to_sec(velo_duration_mmss),
    velo_max_load_watt = as.numeric(velo_max_load_watt),
    velo_max_hr_bpm = as.numeric(velo_max_hr_bpm)
  ) %>%
  select(-velo_duration_mmss)

## split every mean±std gait/balance column into <name>_mean and <name>_sd
for (col in mean_sd_cols) {
  split_res <- split_mean_sd(subject_info[[col]])
  subject_info[[paste0(col, "_mean")]] <- split_res$mean
  subject_info[[paste0(col, "_sd")]] <- split_res$sd
  subject_info[[col]] <- NULL
}

## --- documented dataset caveats (PhysioNet "Usage Notes") -----------------
## - patient 073: NYHA class not provided (already NA from source "?")
## - patient 254: gait analysis results "somehow erroneous" per data authors
## - patients 203, 250: did not perform gait analysis (dizziness) -> already all-NA
## - patient 269: did not perform veloergometry -> already all-NA
## - patient 318: stride length and stride time not provided -> already NA
subject_info <- subject_info %>%
  mutate(gait_flagged_erroneous = patient_id == "254")

write_csv(subject_info, "data_derived/subject_info_clean.csv")
saveRDS(subject_info, "data_derived/subject_info_clean.rds")

## quick sanity checks
stopifnot(nrow(subject_info) == 80)
print(table(subject_info$surgery_type))
print(summary(subject_info$efs_score))
print(colSums(is.na(subject_info)))
