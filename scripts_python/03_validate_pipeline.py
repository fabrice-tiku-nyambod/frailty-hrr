"""
Visual + numeric validation of the R-peak/SQI pipeline and the
accelerometer stillness detector, on a handful of records, focused on
the TUG test (short, has a recorded duration -> naive comparator available).
"""
import numpy as np
import pandas as pd
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from signal_utils import (
    detect_rpeaks, rr_intervals_and_hr, ecg_signal_quality,
    signal_vector_magnitude, movement_energy, quiet_baseline_threshold,
    find_stillness_onset,
)

BASE_DIR = "../extracted/wearable-based-signals-during-physical-exercises-from-patients-with-frailty-after-open-heart-surgery-1.0.0"

# patients where session 1 contains all 5 tests (from test-availability.csv)
PATIENTS = ["001", "048", "061"]

# --- pull TUG recorded time (s) directly from subject-info.csv for the naive comparator
subject_info = pd.read_csv(f"{BASE_DIR}/subject-info.csv", skiprows=2, header=None)
col_names = ["patient_id", "age", "gender", "height", "weight", "efs", "days_after_surgery",
             "surgery_type", "nyha", "af", "copd", "depression", "musculoskeletal",
             "oncological", "ace_inhibitors", "beta_blockers", "ccb",
             "mwt6_distance", "tug_time_s"] + [f"col{i}" for i in range(19, 43)]
subject_info.columns = col_names
subject_info["patient_id"] = subject_info["patient_id"].astype(str).str.zfill(3)
tug_time_lookup = subject_info.set_index("patient_id")["tug_time_s"].to_dict()


def run_patient(patient_id, session=1):
    base = f"{BASE_DIR}/acc/{patient_id}_{session}_acc"
    ecg_base = f"{BASE_DIR}/ecg/{patient_id}_{session}_ecg"

    acc_rec = wfdb.rdrecord(base)
    ecg_rec = wfdb.rdrecord(ecg_base)
    ann = wfdb.rdann(base, 'atr')

    fs_acc = acc_rec.fs
    fs_ecg = ecg_rec.fs

    tug_sample = None
    for s, label in zip(ann.sample, ann.aux_note):
        if label == "TUG":
            tug_sample = s
            break
    if tug_sample is None:
        print(f"[{patient_id}] no TUG annotation found, skipping")
        return

    tug_onset_sec = tug_sample / fs_acc
    tug_recorded_s = float(tug_time_lookup.get(patient_id, np.nan))
    naive_end_sec = tug_onset_sec + tug_recorded_s if not np.isnan(tug_recorded_s) else tug_onset_sec + 30

    # analysis window: 3.5 min before onset -> 4 min after naive end
    win_start_sec = max(0, tug_onset_sec - 210)
    win_end_sec = naive_end_sec + 240

    # --- ACC processing over the window
    svm = signal_vector_magnitude(acc_rec.p_signal)
    me = movement_energy(svm, fs_acc, window_sec=2.0)

    quiet_start = int((tug_onset_sec - 180) * fs_acc)
    quiet_end = int((tug_onset_sec - 20) * fs_acc)
    threshold = quiet_baseline_threshold(me, fs_acc, quiet_start, quiet_end)

    naive_end_sample = int(naive_end_sec * fs_acc)
    still_sample = find_stillness_onset(me, fs_acc, naive_end_sample, threshold,
                                         min_still_duration_sec=15.0)

    # --- ECG processing over the window (restrict to window to keep XQRS fast)
    ecg_lo = int(win_start_sec * fs_ecg)
    ecg_hi = int(win_end_sec * fs_ecg)
    ecg_segment = ecg_rec.p_signal[ecg_lo:ecg_hi, 0]
    qrs_local = detect_rpeaks(ecg_segment, fs_ecg)
    qrs_global = qrs_local + ecg_lo

    mid_samples, hr = rr_intervals_and_hr(qrs_global, fs_ecg)
    good, corr = ecg_signal_quality(ecg_segment, qrs_local, fs_ecg)
    good = good[1:]  # align to rr/hr length (drop first beat, no preceding interval)

    hr_good = hr.copy()
    hr_good[~good] = np.nan
    t_hr = mid_samples / fs_ecg

    # baseline HR: mean of good-quality HR in last 60s of the pre-test quiet window
    baseline_mask = (t_hr >= tug_onset_sec - 80) & (t_hr <= tug_onset_sec - 20) & good
    baseline_hr = np.nanmean(hr[baseline_mask]) if baseline_mask.any() else np.nan

    def hr_at(t0, window=60):
        mask = (t_hr >= t0) & (t_hr <= t0 + window) & good
        return np.nanmean(hr[mask]) if mask.any() else np.nan

    naive_recovery_hr = hr_at(naive_end_sec)
    confirmed_recovery_hr = hr_at(still_sample / fs_acc) if still_sample is not None else np.nan

    print(f"\n=== Patient {patient_id} ===")
    print(f"TUG onset: {tug_onset_sec:.1f}s | recorded TUG time: {tug_recorded_s:.1f}s")
    print(f"naive end: {naive_end_sec:.1f}s | confirmed stillness onset: "
          f"{still_sample/fs_acc:.1f}s" if still_sample is not None else "confirmed stillness: NOT FOUND")
    if still_sample is not None:
        print(f"discrepancy (confirmed - naive): {still_sample/fs_acc - naive_end_sec:.1f}s")
    print(f"baseline HR: {baseline_hr:.1f} bpm")
    print(f"HR 60s after naive end: {naive_recovery_hr:.1f} bpm "
          f"(naive residual excess = {naive_recovery_hr - baseline_hr:.1f})")
    if still_sample is not None:
        print(f"HR 60s after confirmed stillness: {confirmed_recovery_hr:.1f} bpm "
              f"(confirmed residual excess = {confirmed_recovery_hr - baseline_hr:.1f})")
    print(f"beats detected: {len(qrs_global)} | good-quality fraction: {np.mean(good):.2f}")

    # --- plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t_me = np.arange(len(me)) / fs_acc
    win_mask = (t_me >= win_start_sec) & (t_me <= win_end_sec)
    axes[0].plot(t_me[win_mask], me[win_mask], color='steelblue', linewidth=0.6)
    axes[0].axhline(threshold, color='gray', linestyle=':', label='quiet threshold')
    axes[0].axvline(tug_onset_sec, color='green', linestyle='--', label='TUG onset (annotation)')
    axes[0].axvline(naive_end_sec, color='orange', linestyle='--', label='naive end (annotation+duration)')
    if still_sample is not None:
        axes[0].axvline(still_sample / fs_acc, color='red', linestyle='--', label='confirmed stillness onset')
    axes[0].set_ylabel('ACC movement energy (g)')
    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].set_title(f'Patient {patient_id} — TUG test window')

    win_mask_hr = (t_hr >= win_start_sec) & (t_hr <= win_end_sec)
    axes[1].plot(t_hr[win_mask_hr], hr[win_mask_hr], color='lightcoral', linewidth=0.5, alpha=0.5, label='all beats')
    axes[1].plot(t_hr[win_mask_hr & good], hr[win_mask_hr & good], color='firebrick', linewidth=1.0, label='good-quality')
    axes[1].axvline(tug_onset_sec, color='green', linestyle='--')
    axes[1].axvline(naive_end_sec, color='orange', linestyle='--')
    if still_sample is not None:
        axes[1].axvline(still_sample / fs_acc, color='red', linestyle='--')
    axes[1].set_ylabel('Heart rate (bpm)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    out_path = f"../data_derived/validate_{patient_id}_TUG.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved plot: {out_path}")


if __name__ == "__main__":
    for pid in PATIENTS:
        run_patient(pid)
