"""
Directly test the central load-bearing assumption behind RQ1: that the naive-
vs-confirmed discrepancy reflects genuine ongoing movement during the "naive
rest" window, not merely sampling a later point on an otherwise-uneventful
monotonic HR decay curve (a reviewer's sharpest possible objection).

For every test instance with a computable discrepancy (naive_end_sec and
t_still_sec both defined), we re-derive the accelerometer movement-energy
trace and compute, for the interval [naive_end_sec, t_still_sec]:
  - whether movement energy AT naive_end_sec itself exceeds the patient's own
    quiet threshold (was the patient concretely still moving at the exact
    moment naive timing assumed rest had begun?)
  - the fraction of that interval spent above threshold

If these are high, the naive window was capturing real, ongoing,
above-threshold movement -- not just an earlier sample of an already-still
patient -- which is the direct evidence the manuscript currently lacks.
"""
import numpy as np
import pandas as pd
import wfdb

from signal_utils import signal_vector_magnitude, movement_energy

BASE_DIR = "../extracted/wearable-based-signals-during-physical-exercises-from-patients-with-frailty-after-open-heart-surgery-1.0.0"

hrr = pd.read_csv('../data_derived/hrr_features_all.csv')
hrr['patient_id'] = hrr['patient_id'].astype(str).str.zfill(3)
test_avail = pd.read_csv('../data_derived/test_availability_long.csv', dtype={"patient_id": str})
test_avail['patient_id'] = test_avail['patient_id'].str.zfill(3)

valid = hrr[hrr['discrepancy_sec'].notna()].copy()
print(f"n test instances with valid discrepancy: {len(valid)}")

results = []
for _, row in valid.iterrows():
    pid, session = row['patient_id'], int(row['session'])
    test_type = row['test_type']
    match = test_avail[(test_avail['patient_id'] == pid) & (test_avail['session'] == session) &
                        (test_avail['test_type'] == test_type)]
    if match.empty or pd.isna(match.iloc[0]['record_base']):
        continue
    record_base = match.iloc[0]['record_base']

    try:
        acc_rec = wfdb.rdrecord(f"{BASE_DIR}/acc/{record_base}_acc")
    except Exception as e:
        print(f"  skip {pid}/{test_type}: {e}")
        continue
    fs_acc = acc_rec.fs
    svm = signal_vector_magnitude(acc_rec.p_signal)
    me = movement_energy(svm, fs_acc, window_sec=2.0)

    naive_end_sec = row['naive_end_sec']
    t_still_sec = row['t_still_sec']
    threshold = row['quiet_threshold']

    naive_end_sample = int(round(naive_end_sec * fs_acc))
    still_sample = int(round(t_still_sec * fs_acc))
    if naive_end_sample < 0 or naive_end_sample >= len(me) or still_sample <= naive_end_sample:
        continue

    me_at_naive_end = me[naive_end_sample]
    interval = me[naive_end_sample:still_sample]
    frac_above = float(np.mean(interval > threshold))

    results.append({
        'patient_id': pid, 'test_type': test_type, 'discrepancy_sec': row['discrepancy_sec'],
        'me_at_naive_end': me_at_naive_end, 'threshold': threshold,
        'naive_end_above_threshold': bool(me_at_naive_end > threshold),
        'frac_interval_above_threshold': frac_above,
    })

res_df = pd.DataFrame(results)
res_df.to_csv('../data_derived/discrepancy_motion_verification.csv', index=False)

print(f"\nn analyzed: {len(res_df)}")
print(f"% of instances where movement energy AT naive_end_sec exceeds threshold: "
      f"{100*res_df['naive_end_above_threshold'].mean():.1f}%")
print(f"\nFraction of [naive_end, confirmed_onset] interval spent above threshold:")
print(res_df['frac_interval_above_threshold'].describe())
print(f"\nmedian: {res_df['frac_interval_above_threshold'].median():.3f}")
print(f"% of instances with >50% of interval above threshold: "
      f"{100*(res_df['frac_interval_above_threshold'] > 0.5).mean():.1f}%")
print(f"% of instances with >0% (i.e. ANY) time above threshold: "
      f"{100*(res_df['frac_interval_above_threshold'] > 0).mean():.1f}%")

print("\n--- by test type ---")
print(res_df.groupby('test_type').agg(
    n=('patient_id', 'count'),
    pct_naive_end_above=('naive_end_above_threshold', lambda x: 100*x.mean()),
    median_frac_above=('frac_interval_above_threshold', 'median'),
))
