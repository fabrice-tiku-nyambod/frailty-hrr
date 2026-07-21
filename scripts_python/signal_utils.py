"""
Core signal-processing utilities for the motion-verified HRR analysis.

Two independent pipelines:
  - ECG: R-peak detection + a template-correlation signal-quality index (SQI)
    to flag/exclude beats and windows too noisy to trust.
  - ACC: signal vector magnitude + movement-energy + patient-specific
    stillness (motion-cessation) detection.

Both operate on numpy arrays and sample indices; callers handle WFDB I/O.
"""
import os
import subprocess
import sys
import tempfile
import numpy as np
from scipy.ndimage import median_filter
import wfdb
import wfdb.processing


# --------------------------------------------------------------------------
# ECG: R-peak detection
# --------------------------------------------------------------------------

def detect_rpeaks(ecg_signal, fs):
    """Detect R-peak sample indices using the WFDB XQRS detector."""
    return wfdb.processing.xqrs_detect(sig=ecg_signal, fs=fs, verbose=False)


_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "_xqrs_subprocess_worker.py")


def detect_rpeaks_with_timeout(ecg_signal, fs, timeout_sec=60):
    """
    Runs the WFDB XQRS detector in a genuine OS-level subprocess (via
    subprocess.run, not multiprocessing) with a hard wall-clock timeout.
    Necessary because piloting on this dataset found at least one severely
    degraded ECG recording (amplitude/noise far outside the range of every
    other record, consistent with the dataset's own documented caveat that
    "some ECG signals might be unsuitable for analysis due to poor
    quality") on which the detector never returns, with no exception
    raised -- a plain try/except cannot guard against this, only a hard
    timeout can. An earlier multiprocessing.Process/Queue-based version of
    this timeout was found to be unreliable in this environment (timed out
    even on known-good records) and was replaced with this subprocess.run
    version for that reason. Returns None on timeout or error; callers
    should treat that record/test as excluded for signal-quality reasons,
    not retry.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "sig.npy")
        out_path = os.path.join(tmpdir, "qrs.npy")
        np.save(in_path, np.asarray(ecg_signal))
        try:
            subprocess.run(
                [sys.executable, _WORKER_SCRIPT, in_path, out_path, str(fs)],
                timeout=timeout_sec, capture_output=True, check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return None
        if not os.path.exists(out_path):
            return None
        return np.load(out_path)


def rr_intervals_and_hr(qrs_indices, fs):
    """Instantaneous HR (bpm) at the midpoint sample of each RR interval."""
    qrs_indices = np.asarray(qrs_indices)
    rr_sec = np.diff(qrs_indices) / fs
    hr = 60.0 / rr_sec
    midpoint_samples = (qrs_indices[1:] + qrs_indices[:-1]) / 2.0
    return midpoint_samples, hr


# --------------------------------------------------------------------------
# ECG: signal-quality index (template-correlation based)
# --------------------------------------------------------------------------

def _beat_template_correlation(ecg_signal, qrs_indices, fs, half_win_sec=0.25,
                                segment_sec=10, min_beats_per_segment=3,
                                max_lag_sec=0.06):
    """
    For each ~segment_sec-long non-overlapping window, build a median beat
    template from the beats it contains (using a wider extraction window
    than the correlation window, so a small lag search stays in-bounds),
    and correlate every beat in that window against the template allowing
    a small alignment lag. Returns, per beat, its best-lag correlation to
    the local template (NaN where the containing segment had too few
    beats to build a reliable template).

    Vectorized across beats-within-a-segment and across lag values (only
    the outer loop over segments and the small lag-search range remain as
    Python-level loops); an earlier per-beat Python loop implementation was
    found to be the dominant cost of the full pipeline (multi-hour runtime
    across the dataset) and was replaced with this version for that reason.
    """
    half_win = int(round(half_win_sec * fs))
    max_lag = max(1, int(round(max_lag_sec * fs)))
    extract_half_win = half_win + max_lag
    wide_len = 2 * extract_half_win
    template_len = 2 * half_win
    n = len(ecg_signal)
    qrs_indices = np.asarray(qrs_indices)

    corr = np.full(len(qrs_indices), np.nan)
    seg_len = int(round(segment_sec * fs))
    if seg_len <= 0:
        return corr

    seg_id = qrs_indices // seg_len
    offsets = np.arange(wide_len)
    for seg in np.unique(seg_id):
        beat_idx = np.where(seg_id == seg)[0]
        r = qrs_indices[beat_idx]
        lo = r - extract_half_win
        hi = r + extract_half_win
        valid = (lo >= 0) & (hi <= n)
        beat_idx = beat_idx[valid]
        lo = lo[valid]
        if len(beat_idx) < min_beats_per_segment:
            continue

        # vectorized extraction: one row per beat, wide_len samples each
        idx_matrix = lo[:, None] + offsets[None, :]
        wide_waveforms = ecg_signal[idx_matrix]

        central = wide_waveforms[:, max_lag:max_lag + template_len]
        template = np.median(central, axis=0)
        template_centered = template - template.mean()
        template_norm = np.linalg.norm(template_centered)
        if template_norm == 0:
            continue

        best_corr = np.full(len(beat_idx), -1.0)
        for lag in range(-max_lag, max_lag + 1):
            start = max_lag + lag
            w = wide_waveforms[:, start:start + template_len]
            w_centered = w - w.mean(axis=1, keepdims=True)
            w_norm = np.linalg.norm(w_centered, axis=1)
            ok = w_norm > 0
            c = np.full(len(beat_idx), -1.0)
            c[ok] = (w_centered[ok] @ template_centered) / (w_norm[ok] * template_norm)
            best_corr = np.maximum(best_corr, c)

        corr[beat_idx] = best_corr
    return corr


def rr_outlier_mask(rr_sec, rel_threshold=0.2, median_window=9):
    """
    Flags RR intervals that deviate from their local (rolling-median)
    neighborhood by more than rel_threshold (fractional). Catches ectopic
    beats / missed-or-extra detections that produce a short-RR/long-RR
    pair even when the beat morphology itself still looks plausible
    (a single-detector template-correlation check alone misses these).
    """
    rr_sec = np.asarray(rr_sec, dtype=float)
    if len(rr_sec) < 3:
        return np.zeros(len(rr_sec), dtype=bool)
    local_median = median_filter(rr_sec, size=median_window, mode='nearest')
    rel_dev = np.abs(rr_sec - local_median) / local_median
    return rel_dev > rel_threshold


def ecg_signal_quality(ecg_signal, qrs_indices, fs,
                        min_hr=30, max_hr=220, corr_threshold=0.8,
                        rr_rel_threshold=0.2):
    """
    Returns a per-beat boolean 'good' mask combining:
      - physiologically plausible surrounding RR interval(s)
      - RR-interval deviation from local neighborhood (ectopic/missed-beat filter)
      - template-correlation above corr_threshold
    A beat with no valid correlation (edge of a sparse segment) is treated
    as not-good (conservative).
    """
    qrs_indices = np.asarray(qrs_indices)
    n_beats = len(qrs_indices)
    good = np.ones(n_beats, dtype=bool)

    if n_beats >= 2:
        rr_sec = np.diff(qrs_indices) / fs
        hr = 60.0 / rr_sec
        implausible = (hr < min_hr) | (hr > max_hr)
        rr_outlier = rr_outlier_mask(rr_sec, rel_threshold=rr_rel_threshold)
        bad_rr = implausible | rr_outlier
        # a bad RR interval taints both beats bounding it
        bad_beats = np.zeros(n_beats, dtype=bool)
        bad_beats[:-1] |= bad_rr
        bad_beats[1:] |= bad_rr
        good &= ~bad_beats

    corr = _beat_template_correlation(ecg_signal, qrs_indices, fs)
    good &= (corr >= corr_threshold)

    return good, corr


# --------------------------------------------------------------------------
# ACC: movement energy + stillness detection
# --------------------------------------------------------------------------

def signal_vector_magnitude(acc_signal):
    """Orientation-invariant magnitude; sidesteps chest-strap inversion issue."""
    return np.sqrt((acc_signal ** 2).sum(axis=1))


def movement_energy(svm, fs, window_sec=2.0):
    """Moving standard deviation of the SVM signal over `window_sec` windows."""
    win = max(1, int(round(window_sec * fs)))
    kernel = np.ones(win) / win
    mean = np.convolve(svm, kernel, mode='same')
    mean_sq = np.convolve(svm ** 2, kernel, mode='same')
    var = np.clip(mean_sq - mean ** 2, a_min=0, a_max=None)
    return np.sqrt(var)


def quiet_baseline_threshold(me, fs, quiet_start_sample, quiet_end_sample, margin_sd=3.0):
    """
    Patient/session-specific 'what does stillness look like' threshold,
    estimated from a known quiet period (e.g., tail of the pre-test rest).
    """
    segment = me[quiet_start_sample:quiet_end_sample]
    if len(segment) == 0:
        return None
    return float(segment.mean() + margin_sd * segment.std())


def find_stillness_onset(me, fs, search_start_sample, threshold,
                          min_still_duration_sec=90.0, max_search_sec=600.0):
    """
    First sample index at/after search_start_sample where `me` drops below
    `threshold` and stays below it continuously for min_still_duration_sec.
    Returns None if no such point is found within max_search_sec.

    min_still_duration_sec default is deliberately close to (but somewhat
    below) the study protocol's mandated >=3-minute post-test rest period,
    not an arbitrary value: a short threshold (e.g. 15s) was found during
    piloting to mistake mid-test pauses -- e.g. a frail patient stopping to
    catch their breath partway through the 6MWT -- for genuine post-test
    recovery onset, since the patient briefly holds still before resuming
    exercise. Because the run-length count resets on any renewed movement,
    raising this duration requirement is sufficient on its own to reject
    such pauses while still being satisfied by genuine (much longer,
    protocol-mandated) rest periods; no separate verification step is
    needed. Sensitivity to this choice should be checked (see manuscript
    plan, Section 6).
    """
    min_still_samples = int(round(min_still_duration_sec * fs))
    max_search_samples = int(round(max_search_sec * fs))
    search_end = min(len(me), search_start_sample + max_search_samples)

    below = (me[search_start_sample:search_end] < threshold).astype(np.int32)
    if len(below) < min_still_samples:
        return None

    # vectorized sliding-window sum via cumulative sum: window_sums[i] is the
    # count of "below-threshold" samples in below[i : i+min_still_samples].
    # A window is a valid run start iff every sample in it is below
    # threshold, i.e. window_sums[i] == min_still_samples. Replaces an
    # earlier pure-Python per-sample loop (a major contributor to a
    # multi-hour full-dataset runtime) with an O(n) vectorized equivalent.
    cumsum = np.cumsum(np.concatenate(([0], below)))
    window_sums = cumsum[min_still_samples:] - cumsum[:-min_still_samples]
    candidates = np.flatnonzero(window_sums == min_still_samples)
    if len(candidates) == 0:
        return None
    return search_start_sample + int(candidates[0])
