"""
Standalone worker invoked as a real OS subprocess (not multiprocessing) so
the caller can enforce a hard wall-clock timeout via subprocess.run(...,
timeout=...) -- see detect_rpeaks_with_timeout in signal_utils.py for why
this is necessary (the WFDB XQRS detector was found to hang indefinitely,
with no exception raised, on at least one severely degraded ECG recording).
"""
import sys
import numpy as np
import wfdb.processing

if __name__ == "__main__":
    in_path, out_path, fs_str = sys.argv[1], sys.argv[2], sys.argv[3]
    sig = np.load(in_path)
    qrs = wfdb.processing.xqrs_detect(sig=sig, fs=float(fs_str), verbose=False)
    np.save(out_path, qrs)
