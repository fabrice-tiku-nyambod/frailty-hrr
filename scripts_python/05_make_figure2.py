"""
Publication-quality Figure 2: example signal traces illustrating naive vs.
accelerometer-confirmed rest-onset timing for patient 208's VELO test instance
(a clean, dramatic, VERIFIED-from-hrr_features_all.csv case: 97.2 s
naive-vs-confirmed discrepancy; naive timing shows 24.5 bpm residual excess
HR vs. 5.4 bpm confirmed; good_fraction_window 0.994). An earlier version of
this figure used patient 048's TUG instance based on a stale, pre-90s-
threshold-fix number (50 s discrepancy, 0.1 bpm confirmed) that no longer
matches the final pipeline output (376.7 s, 4.4 bpm) -- always re-verify the
specific example against the authoritative CSV before finalizing a figure.

Palette: validated categorical slots from the dataviz skill's reference
palette (references/palette.md) -- blue (#2a78d6) for the primary/confirmed
signal, orange (#eb6834) for the naive/assumption marker, muted gray for
de-emphasized context (raw beats, threshold, test onset). The automated
CVD-safety validator (scripts/validate_palette.js) could not be run in this
environment (node not installed); the blue/aqua pair used here is a
documented adjacent pair in the validated ordering (worst-case adjacent CVD
delta-E 24.2, well above the >=12 target), and blue-orange are near-
complementary hues that remain robustly distinguishable under the common
red-green CVD types -- noted as a reasoned substitute for the script, not
a full replacement for it.
"""
import numpy as np
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from signal_utils import (
    detect_rpeaks_with_timeout, rr_intervals_and_hr, ecg_signal_quality,
    signal_vector_magnitude, movement_energy, quiet_baseline_threshold,
)

BASE_DIR = "../extracted/wearable-based-signals-during-physical-exercises-from-patients-with-frailty-after-open-heart-surgery-1.0.0"

# palette (dataviz skill reference palette, validated categorical slots)
COLOR_CONFIRMED = "#2a78d6"   # slot 1, blue
COLOR_NAIVE = "#eb6834"       # slot 8, orange
COLOR_GOOD_HR = "#0b0b0b"     # primary ink
COLOR_ALL_HR = "#c3c2b7"      # muted/baseline gray
COLOR_THRESH = "#898781"      # muted axis gray
COLOR_ONSET = "#52514e"       # secondary ink

PATIENT, SESSION = "208", "1"
TEST_LABEL = "VELO"
TEST_ONSET_SEC = 4304.995
NAIVE_END_SEC = 4403.995  # onset + recorded veloergometry duration

WIN_START = TEST_ONSET_SEC - 130
WIN_END = 4501.185 + 220  # confirmed onset (re-derived below) + buffer

acc_rec = wfdb.rdrecord(f"{BASE_DIR}/acc/{PATIENT}_{SESSION}_acc")
ecg_rec = wfdb.rdrecord(f"{BASE_DIR}/ecg/{PATIENT}_{SESSION}_ecg")
fs_acc, fs_ecg = acc_rec.fs, ecg_rec.fs

svm = signal_vector_magnitude(acc_rec.p_signal)
me = movement_energy(svm, fs_acc, window_sec=2.0)
t_me = np.arange(len(me)) / fs_acc

quiet_start = int((TEST_ONSET_SEC - 150) * fs_acc)
quiet_end = int((TEST_ONSET_SEC - 30) * fs_acc)
threshold = quiet_baseline_threshold(me, fs_acc, quiet_start, quiet_end)

ecg_lo = int(WIN_START * fs_ecg)
ecg_hi = int(WIN_END * fs_ecg)
ecg_segment = ecg_rec.p_signal[ecg_lo:ecg_hi, 0]
qrs = detect_rpeaks_with_timeout(ecg_segment, fs_ecg, timeout_sec=60)
mid_samples, hr = rr_intervals_and_hr(qrs, fs_ecg)
good, _ = ecg_signal_quality(ecg_segment, qrs, fs_ecg)
good = good[1:]
t_hr = mid_samples / fs_ecg + WIN_START

# confirmed stillness onset (re-derive exactly as in the pipeline)
from signal_utils import find_stillness_onset
onset_sample = int(TEST_ONSET_SEC * fs_acc)
still_sample = find_stillness_onset(me, fs_acc, onset_sample, threshold,
                                     min_still_duration_sec=90.0, max_search_sec=600.0)
CONFIRMED_ONSET_SEC = still_sample / fs_acc

# --- figure ------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1.2]})

# time axis relative to test onset, in seconds
t0 = TEST_ONSET_SEC

# --- Panel A: movement energy
ax = axes[0]
mask = (t_me >= WIN_START) & (t_me <= WIN_END)
ax.plot(t_me[mask] - t0, me[mask], color=COLOR_CONFIRMED, linewidth=1.0)
ax.axhline(threshold, color=COLOR_THRESH, linestyle=':', linewidth=1.0)
ax.text(WIN_END - t0 - 5, threshold + 3, 'quiet threshold', ha='right', va='bottom',
        fontsize=8, color=COLOR_THRESH)
ax.set_ylabel('Movement energy (g)')
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('A', loc='left', fontweight='bold', fontsize=11)

# --- Panel B: heart rate
ax = axes[1]
mask_hr = (t_hr >= WIN_START) & (t_hr <= WIN_END)
ax.plot(t_hr[mask_hr] - t0, hr[mask_hr], color=COLOR_ALL_HR, linewidth=0.8,
        alpha=0.7, label='All detected beats')
ax.plot(t_hr[mask_hr & good] - t0, hr[mask_hr & good], color=COLOR_GOOD_HR,
        linewidth=1.3, label='Good-quality beats')
ax.set_ylabel('Heart rate (bpm)')
ax.set_xlabel(f'Time relative to {TEST_LABEL} onset (s)')
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('B', loc='left', fontweight='bold', fontsize=11)

# --- shared reference lines. Direct rotated inline labels were tried first
# but overlapped badly since the onset and naive-rest-onset lines are only
# ~100 s apart on a ~550 s wide axis, right where the movement-energy trace
# also peaks -- a small legend reads far more cleanly than fighting label
# placement in that crowded region.
from matplotlib.lines import Line2D
for ax in axes:
    ax.axvline(0, color=COLOR_ONSET, linestyle='--', linewidth=1.0)
    ax.axvline(NAIVE_END_SEC - t0, color=COLOR_NAIVE, linestyle='--', linewidth=1.3)
    ax.axvline(CONFIRMED_ONSET_SEC - t0, color=COLOR_CONFIRMED, linestyle='--', linewidth=1.3)

ref_line_handles = [
    Line2D([0], [0], color=COLOR_ONSET, linestyle='--', linewidth=1.0, label=f'{TEST_LABEL} onset'),
    Line2D([0], [0], color=COLOR_NAIVE, linestyle='--', linewidth=1.3, label='Naive rest onset'),
    Line2D([0], [0], color=COLOR_CONFIRMED, linestyle='--', linewidth=1.3, label='Confirmed rest onset'),
]
axes[0].legend(handles=ref_line_handles, loc='upper right', fontsize=8, frameon=False)
# Panel B's heart-rate trace is dense/spiky throughout, so unlike Panel A
# there is no genuinely empty region to place a frameless legend -- give it
# a solid backing instead of hunting for a placement that doesn't exist.
axes[1].legend(loc='lower right', fontsize=8, frameon=True, facecolor='white',
                edgecolor='none', framealpha=0.92)

fig.suptitle(f'Naive vs. accelerometer-confirmed rest onset (Patient {PATIENT}, {TEST_LABEL} test)',
             fontsize=11, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('../data_derived/figure2_naive_vs_confirmed_example.png', dpi=600)
print('saved: data_derived/figure2_naive_vs_confirmed_example.png')
print(f'naive rest onset: {NAIVE_END_SEC:.1f}s | confirmed: {CONFIRMED_ONSET_SEC:.1f}s | '
      f'discrepancy: {CONFIRMED_ONSET_SEC - NAIVE_END_SEC:.1f}s')
