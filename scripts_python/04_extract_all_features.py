"""
Full-scale extraction: for every available test instance (STAIR, 6MWT, TUG,
VELO; GAIT_ANALYSIS excluded per plan), compute:
  - pre-test quiet-baseline HR and movement-energy threshold
  - accelerometer-confirmed stillness onset (post-test)
  - motion-confirmed HRR60 / residual excess HR (primary outcome)
  - naive/protocol-based comparator, where definable (see manuscript_plan.md 5.4):
      VELO, TUG: onset + recorded duration
      6MWT:      onset + fixed 360 s (protocol-nominal 6 minutes)
      STAIR:     no naive comparator (left NaN)

Processes ECG/ACC once per unique record (not per test) since a single
session recording often contains multiple tests, to avoid redundant
R-peak detection on overlapping windows.
"""
import numpy as np
import pandas as pd
import wfdb

from signal_utils import (
    detect_rpeaks_with_timeout, rr_intervals_and_hr, ecg_signal_quality,
    signal_vector_magnitude, movement_energy, quiet_baseline_threshold,
    find_stillness_onset,
)

RPEAK_TIMEOUT_SEC = 60  # see signal_utils.detect_rpeaks_with_timeout docstring:
                        # guards against the XQRS detector hanging indefinitely
                        # on severely degraded ECG recordings (found empirically
                        # on record 040_1)

BASE_DIR = "../extracted/wearable-based-signals-during-physical-exercises-from-patients-with-frailty-after-open-heart-surgery-1.0.0"
PRIMARY_TESTS = ["STAIR", "6MWT", "TUG", "VELO"]
PRE_QUIET_LOOKBACK_START = 150  # seconds before onset
PRE_QUIET_LOOKBACK_END = 30     # seconds before onset
MAX_SEARCH_SEC = 600            # how far past onset to search for stillness
MIN_STILL_DURATION_SEC = 90.0   # see signal_utils.find_stillness_onset docstring: guards
                                 # against mid-test pauses (e.g. 6MWT breath-catching)
                                 # being mistaken for genuine post-test recovery onset
RECOVERY_WINDOW_SEC = 60
MAX_PLAUSIBLE_RESTING_HR = 120  # bpm; guards against a contaminated baseline window
                                # (found empirically: patients 102/STAIR, 318/STAIR
                                # had baseline_hr of 167.8/193.1 bpm from a low-quality,
                                # low-beat-count window that individually-plausible-range
                                # noisy beats slipped through the per-beat SQI to produce --
                                # no genuine resting HR reaches this range in this cohort
                                # (max otherwise ~110 bpm; median 74, IQR 65-83)


def load_lookup_tables():
    test_avail = pd.read_csv("../data_derived/test_availability_long.csv", dtype={"patient_id": str})
    # patient_id and session were auto-inferred as int/float on read, dropping
    # zero-padding (e.g. "001" -> 1) and turning session into "1.0" -- use the
    # already-correct record_base/ecg_record/acc_record string columns instead
    # of reconstructing filenames from patient_id+session.
    test_avail["patient_id"] = test_avail["patient_id"].str.zfill(3)
    subject_info = pd.read_csv("../data_derived/subject_info_clean.csv")
    subject_info = subject_info[subject_info["patient_id"] != "Patient ID"].copy()
    subject_info["patient_id"] = subject_info["patient_id"].astype(str).str.zfill(3)
    return test_avail, subject_info


def hr_mean_in_window(t_hr, hr, good, t0, t1):
    mask = (t_hr >= t0) & (t_hr <= t1) & good
    if not mask.any():
        return np.nan, 0
    return float(np.nanmean(hr[mask])), int(mask.sum())


def process_record(patient_id, session, record_base, tests_in_record, subject_row):
    acc_base = f"{BASE_DIR}/acc/{record_base}_acc"
    ecg_base = f"{BASE_DIR}/ecg/{record_base}_ecg"

    acc_rec = wfdb.rdrecord(acc_base)
    ecg_rec = wfdb.rdrecord(ecg_base)
    ann = wfdb.rdann(acc_base, 'atr')
    fs_acc, fs_ecg = acc_rec.fs, ecg_rec.fs

    onset_by_test = {}
    for s, label in zip(ann.sample, ann.aux_note):
        if label in PRIMARY_TESTS:
            onset_by_test[label] = s / fs_acc

    # --- whole-record ACC processing (once; this was never the bottleneck
    # and stillness detection genuinely needs the full timeline to search)
    svm = signal_vector_magnitude(acc_rec.p_signal)
    me = movement_energy(svm, fs_acc, window_sec=2.0)

    ecg_full = ecg_rec.p_signal[:, 0]
    n_ecg = len(ecg_full)

    rows = []
    for test_type in tests_in_record:
        if test_type not in onset_by_test:
            continue
        onset_sec = onset_by_test[test_type]

        # pre-test quiet baseline window
        q_start = max(0.0, onset_sec - PRE_QUIET_LOOKBACK_START)
        q_end = max(0.0, onset_sec - PRE_QUIET_LOOKBACK_END)
        q_start_sample, q_end_sample = int(q_start * fs_acc), int(q_end * fs_acc)
        threshold = quiet_baseline_threshold(me, fs_acc, q_start_sample, q_end_sample)

        # confirmed stillness: search starts at test onset itself (the
        # detector naturally skips the active-movement phase since ME
        # stays above threshold until real movement stops)
        onset_sample = int(onset_sec * fs_acc)
        still_sample = None
        if threshold is not None:
            still_sample = find_stillness_onset(
                me, fs_acc, onset_sample, threshold,
                min_still_duration_sec=MIN_STILL_DURATION_SEC,
                max_search_sec=MAX_SEARCH_SEC,
            )
        t_still = still_sample / fs_acc if still_sample is not None else np.nan

        # --- ECG processed only within this test's own window, not the
        # whole recording. Piloting found that running R-peak detection on
        # an entire multi-hour session could time out due to a problem
        # located anywhere in that session (e.g. during an unrelated later
        # test), which then discarded otherwise-usable data for every test
        # in that recording. Restricting to a bounded window around each
        # test's own onset avoids this: it covers the pre-test baseline
        # through the full stillness-search range plus recovery, for
        # *this* test only.
        ecg_win_start = max(0.0, onset_sec - PRE_QUIET_LOOKBACK_START - 10)
        ecg_win_end = onset_sec + MAX_SEARCH_SEC + RECOVERY_WINDOW_SEC + 10
        ecg_lo = int(ecg_win_start * fs_ecg)
        ecg_hi = min(n_ecg, int(ecg_win_end * fs_ecg))
        ecg_segment = ecg_full[ecg_lo:ecg_hi]

        qrs_result = detect_rpeaks_with_timeout(ecg_segment, fs_ecg,
                                                 timeout_sec=RPEAK_TIMEOUT_SEC)
        ecg_excluded = qrs_result is None
        if ecg_excluded:
            hr = np.array([])
            good = np.array([], dtype=bool)
            t_hr = np.array([])
        else:
            mid_samples, hr = rr_intervals_and_hr(qrs_result, fs_ecg)
            good, _ = ecg_signal_quality(ecg_segment, qrs_result, fs_ecg)
            good = good[1:]  # align to rr/hr length
            t_hr = mid_samples / fs_ecg + ecg_win_start  # back to record-global time
        qrs = qrs_result if qrs_result is not None else np.array([], dtype=int)

        baseline_hr, baseline_n = hr_mean_in_window(t_hr, hr, good, q_start, q_end)
        baseline_implausible = (not np.isnan(baseline_hr)) and baseline_hr > MAX_PLAUSIBLE_RESTING_HR
        if baseline_implausible:
            # a contaminated baseline window (low beat count / low quality
            # fraction letting individually-plausible-range but collectively
            # implausible beats through the per-beat SQI) invalidates any
            # outcome referenced to it; NaN propagates through the
            # delta_hr60_* subtractions below automatically
            baseline_hr = np.nan

        confirmed_hr60, confirmed_n = (np.nan, 0)
        if not np.isnan(t_still):
            confirmed_hr60, confirmed_n = hr_mean_in_window(
                t_hr, hr, good, t_still, t_still + RECOVERY_WINDOW_SEC)
        delta_hr60_confirmed = confirmed_hr60 - baseline_hr if not np.isnan(confirmed_hr60) else np.nan

        # naive comparator (test-type-dependent; see manuscript_plan.md 5.4)
        naive_end = np.nan
        if test_type in ("TUG", "VELO"):
            dur_col = "tug_time_s" if test_type == "TUG" else "velo_duration_s"
            dur = subject_row.get(dur_col, np.nan)
            if pd.notna(dur):
                naive_end = onset_sec + float(dur)
        elif test_type == "6MWT":
            naive_end = onset_sec + 360.0
        # STAIR: naive_end stays NaN by design

        naive_hr60, naive_n = (np.nan, 0)
        if not np.isnan(naive_end):
            naive_hr60, naive_n = hr_mean_in_window(
                t_hr, hr, good, naive_end, naive_end + RECOVERY_WINDOW_SEC)
        delta_hr60_naive = naive_hr60 - baseline_hr if not np.isnan(naive_hr60) else np.nan

        discrepancy_sec = (t_still - naive_end) if (not np.isnan(t_still) and not np.isnan(naive_end)) else np.nan

        rows.append({
            "patient_id": patient_id, "session": session, "test_type": test_type,
            "onset_sec": onset_sec, "quiet_threshold": threshold,
            "baseline_hr": baseline_hr, "baseline_n_beats": baseline_n,
            "t_still_sec": t_still, "stillness_found": not np.isnan(t_still),
            "confirmed_hr60": confirmed_hr60, "confirmed_n_beats": confirmed_n,
            "delta_hr60_confirmed": delta_hr60_confirmed,
            "naive_end_sec": naive_end, "naive_hr60": naive_hr60, "naive_n_beats": naive_n,
            "delta_hr60_naive": delta_hr60_naive,
            "discrepancy_sec": discrepancy_sec,
            "total_beats_window": len(qrs),
            "good_fraction_window": float(np.mean(good)) if len(good) else np.nan,
            "ecg_excluded": ecg_excluded,
            "baseline_implausible": baseline_implausible,
        })
    return rows


def main():
    import time
    import sys

    test_avail, subject_info = load_lookup_tables()
    test_avail = test_avail[test_avail["test_type"].isin(PRIMARY_TESTS)]
    test_avail = test_avail.dropna(subset=["record_base"])

    subject_info = subject_info.set_index("patient_id")

    all_rows = []
    grouped = test_avail.groupby(["patient_id", "record_base"])
    n_groups = len(grouped)
    t_start = time.time()
    for i, ((patient_id, record_base), group) in enumerate(grouped, 1):
        tests_in_record = group["test_type"].tolist()
        session = group["session"].iloc[0]
        subject_row = subject_info.loc[patient_id] if patient_id in subject_info.index else {}
        t0 = time.time()
        print(f"[{i}/{n_groups}] patient {patient_id} record {record_base}: {tests_in_record} ...",
              end="", flush=True)
        try:
            rows = process_record(patient_id, session, record_base, tests_in_record, subject_row)
            all_rows.extend(rows)
            elapsed = time.time() - t0
            print(f" done in {elapsed:.1f}s", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f" FAILED after {elapsed:.1f}s: {e}", flush=True)

        # incremental save every 5 records so we never lose progress and can
        # inspect partial results while the run is still going
        if i % 5 == 0 or i == n_groups:
            pd.DataFrame(all_rows).to_csv("../data_derived/hrr_features_all.csv", index=False)
            total_elapsed = time.time() - t_start
            print(f"  [checkpoint: {i}/{n_groups} records, {len(all_rows)} rows, "
                  f"{total_elapsed:.0f}s elapsed, {total_elapsed/i:.1f}s/record avg]", flush=True)

    result = pd.DataFrame(all_rows)
    result.to_csv("../data_derived/hrr_features_all.csv", index=False)
    print(f"\nSaved {len(result)} test-instance rows to data_derived/hrr_features_all.csv")
    print("\n--- coverage by test type ---")
    print(result.groupby("test_type").agg(
        n=("patient_id", "count"),
        stillness_found=("stillness_found", "sum"),
        naive_available=("naive_end_sec", lambda x: x.notna().sum()),
    ))


if __name__ == "__main__":
    main()
