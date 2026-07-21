# Motion-Verified Heart Rate Recovery in Frailty After Open-Heart Surgery

Secondary analysis of the PhysioNet dataset ["Wearable-based signals during physical
exercises from patients with frailty after open-heart surgery"](https://doi.org/10.13026/mp8k-7p27)
(Sokas et al., 2022). Verifies post-exercise rest onset with the accelerometer channel
instead of trusting protocol/annotation timing, and quantifies the resulting bias in
heart rate recovery (HRR) measurement, then examines whether the corrected measure
varies with frailty severity and clinical covariates.

See `manuscript_plan.md` for the full research plan (objectives, hypotheses, statistical
plan) and `manuscript_methods_draft.md` for the consolidated Methods writeup including
pipeline validation findings and disclosed limitations. The manuscript itself is
submitted separately and is not part of this repository.

## Data

This repo does **not** include the raw dataset (2.4GB, excluded via `.gitignore`).
Download it from PhysioNet and extract into `extracted/` at the repo root:

```
https://physionet.org/content/wearable-exercises-frailty/1.0.0/
```

After extraction, `extracted/wearable-based-signals-during-physical-exercises-from-patients-with-frailty-after-open-heart-surgery-1.0.0/`
should contain `acc/`, `ecg/`, `subject-info.csv`, and `test-availability.csv`.

## Setup

**Python** (signal processing):
```
pip install -r requirements.txt
```

**R** (clinical data cleaning + statistical modeling): `readr`, `dplyr`, `tidyr`,
`stringr`, `ggplot2`, `lme4`, `lmerTest`, `car`.

## Pipeline order

1. `scripts_r/01_load_clean_subject_info.R` — cleans `subject-info.csv` (two-row
   header, mean±std splitting, decimal-comma fix, factor recoding) →
   `data_derived/subject_info_clean.csv`.
2. `scripts_r/02_load_test_availability.R` — reshapes `test-availability.csv` into
   long form with derived record filenames → `data_derived/test_availability_long.csv`.
3. `scripts_python/04_extract_all_features.py` — the core pipeline: R-peak detection
   and signal-quality screening (`scripts_python/signal_utils.py`), accelerometer
   movement-energy and stillness detection, naive vs. motion-confirmed rest-onset
   timing, and outcome computation for every test instance → `data_derived/hrr_features_all.csv`.
   (`scripts_python/03_validate_pipeline.py` is an optional pilot validation/diagnostic
   script on a handful of patients, useful for sanity-checking changes to `signal_utils.py`.)
4. `scripts_r/03_statistical_analysis.R` — RQ1 (naive vs. motion-confirmed timing:
   Wilcoxon tests, Bland-Altman) and RQ2 (block-wise linear mixed-effects models of
   residual excess heart rate against frailty and clinical covariates). Writes a full
   console log to `data_derived/rq_analysis_log.txt` and the figures/tables underlying
   the main-text model table, the block-comparison supplementary table, the test-type/
   EFS figure, and the residual-diagnostics and raw-EFS supplementary figures.
5. `scripts_python/07_verify_discrepancy_is_motion.py` — direct check that the naive-
   vs-confirmed timing discrepancy reflects genuine accelerometer-measured motion
   rather than an artifact of sampling later on a decaying heart-rate curve →
   `data_derived/discrepancy_motion_verification.csv`.
6. `scripts_python/05_make_figure2.py` — single-instance example signal traces
   (naive vs. motion-confirmed rest onset) for a representative test.
7. `scripts_python/08_make_figure_discrepancy_distribution.py` — population-level
   distribution of the naive-to-confirmed discrepancy and the motion-verification
   result across all test instances.
8. `scripts_python/06_make_flow_diagram.py` — CONSORT-style participant/test-instance
   exclusion flow diagram, with every count re-derived from `hrr_features_all.csv`
   at runtime (not hand-typed).

`scripts_python/_xqrs_subprocess_worker.py` is an internal helper invoked by
`signal_utils.detect_rpeaks_with_timeout` as a real subprocess, so a hard timeout can
be enforced on R-peak detection (needed because the detector was found to hang
indefinitely, with no exception raised, on a small number of degraded recordings).

All figures are saved to `data_derived/` at 600 DPI (gitignored, since they are fully
regenerable from the scripts above plus the derived CSVs already committed).

## Current status

Pipeline complete and statistical analysis finalized. Of 293 available test instances
(80 patients), 211 (72%, 75/80 patients) yielded a usable motion-confirmed outcome
after the full exclusion cascade (ECG/accelerometer coverage gaps, stillness not
located, implausible baseline, uncomputable recovery window — see
`scripts_python/06_make_flow_diagram.py`). Motion-confirmed rest onset occurs
significantly later than protocol-timed onset and corrects a systematic bias in
heart rate recovery estimation; frailty-severity and atrial-fibrillation associations
with the corrected measure are marginal and reported as hypothesis-generating. Full
detail and caveats are in the manuscript (submitted separately).
