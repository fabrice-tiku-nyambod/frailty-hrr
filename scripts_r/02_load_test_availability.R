## 02_load_test_availability.R
## Loads test-availability.csv into long/tidy form: one row per
## patient x test, giving the record base name (e.g. "001_1") used to
## build the actual .dat/.hea/.atr filenames in ecg/ and acc/, or NA if
## that test was not logged/transferred for that patient.

library(readr)
library(dplyr)
library(tidyr)
library(stringr)

avail_path <- "extracted/wearable-based-signals-during-physical-exercises-from-patients-with-frailty-after-open-heart-surgery-1.0.0/test-availability.csv"

test_availability <- read_csv(avail_path, na = "-", show_col_types = FALSE) %>%
  rename(patient_id = `Patient ID`) %>%
  pivot_longer(cols = c(STAIR, `6MWT`, TUG, VELO, GAIT_ANALYSIS),
               names_to = "test_type", values_to = "record_base") %>%
  mutate(
    patient_id = sprintf("%03d", as.integer(patient_id)),
    session = as.integer(str_extract(record_base, "(?<=_)[0-9]+$")),
    ecg_record = if_else(!is.na(record_base), paste0(record_base, "_ecg"), NA_character_),
    acc_record = if_else(!is.na(record_base), paste0(record_base, "_acc"), NA_character_)
  )

write_csv(test_availability, "data_derived/test_availability_long.csv")
saveRDS(test_availability, "data_derived/test_availability_long.rds")

## sanity: how many patients have each test available
print(test_availability %>% count(test_type, available = !is.na(record_base)))
