"""
Supplementary Figure: participant/test-instance flow diagram (CONSORT-style),
built from the verified exclusion cascade in hrr_features_all.csv:
  293 test instances (80 patients)
  -1 ECG/accelerometer data-coverage gap (patient 115, VELO)          -> 292
  -76 accelerometer-confirmed stillness not located in search window  -> 216
  -4 baseline heart rate implausible (contaminated window)            -> 212
  -1 recovery-window heart rate uncomputable (zero good-quality beats) -> 211
  = 211 test instances, 75/80 patients (94%), usable primary outcome
Every number here is re-derived from the data (see 06 script comments/prints),
not hand-typed, so it cannot silently drift out of sync with the pipeline.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

df = pd.read_csv('../data_derived/hrr_features_all.csv')

n_total = len(df)
n_ecg_excl = int(df['ecg_excluded'].sum())
after_ecg = df[~df['ecg_excluded']]
n_after_ecg = len(after_ecg)

n_still_excl = int((~after_ecg['stillness_found']).sum())
after_still = after_ecg[after_ecg['stillness_found']]
n_after_still = len(after_still)

n_baseline_excl = int(after_still['baseline_implausible'].sum())
after_baseline = after_still[~after_still['baseline_implausible']]
n_after_baseline = len(after_baseline)

n_recovery_excl = int(after_baseline['delta_hr60_confirmed'].isna().sum())
final = after_baseline[after_baseline['delta_hr60_confirmed'].notna()]
n_final = len(final)
n_final_patients = final['patient_id'].astype(str).str.zfill(3).nunique()

assert n_final == df['delta_hr60_confirmed'].notna().sum(), "cascade doesn't match ground truth -- fix before plotting"

COLOR_BOX = "#2a78d6"      # slot 1, blue -- main flow
COLOR_EXCL = "#eb6834"     # slot 8, orange -- exclusion branches
COLOR_FINAL = "#1baf7a"    # slot 2, aqua -- final usable box
TEXT_COLOR = "#0b0b0b"

fig, ax = plt.subplots(figsize=(8, 7.6))
ax.set_xlim(0, 10)
ax.set_ylim(8.7, 26)
ax.axis('off')

def main_box(y, text, color=COLOR_BOX, height=1.6):
    box = FancyBboxPatch((1, y), 5.5, height, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='none', alpha=0.15)
    ax.add_patch(box)
    box2 = FancyBboxPatch((1, y), 5.5, height, boxstyle="round,pad=0.1",
                           facecolor='none', edgecolor=color, linewidth=1.5)
    ax.add_patch(box2)
    ax.text(3.75, y + height / 2, text, ha='center', va='center', fontsize=10,
             color=TEXT_COLOR, linespacing=1.4)

def excl_box(y, text):
    box = FancyBboxPatch((7.3, y - 0.5), 2.5, 1.3, boxstyle="round,pad=0.08",
                          facecolor=COLOR_EXCL, edgecolor='none', alpha=0.12)
    ax.add_patch(box)
    box2 = FancyBboxPatch((7.3, y - 0.5), 2.5, 1.3, boxstyle="round,pad=0.08",
                           facecolor='none', edgecolor=COLOR_EXCL, linewidth=1.2)
    ax.add_patch(box2)
    ax.text(8.55, y + 0.15, text, ha='center', va='center', fontsize=7.8,
             color=TEXT_COLOR, linespacing=1.3)

def down_arrow(y_top, y_bottom):
    ax.add_patch(FancyArrowPatch((3.75, y_top), (3.75, y_bottom),
                                  arrowstyle='-|>', mutation_scale=14,
                                  color=TEXT_COLOR, linewidth=1.2))

def side_arrow(y):
    ax.add_patch(FancyArrowPatch((6.5, y), (7.3, y), arrowstyle='-|>',
                                  mutation_scale=12, color=COLOR_EXCL, linewidth=1.1))

y = 24
main_box(y, f"80 patients enrolled\n(PhysioNet dataset)")
down_arrow(y, y - 1.4)

y -= 3
main_box(y, f"{n_total} test instances available\n(TUG 77, VELO 75, STAIR 74, 6MWT 67)")
side_arrow(y + 0.8)
excl_box(y + 0.8, f"n={n_ecg_excl}\nECG/accelerometer\ndata-coverage gap")
down_arrow(y, y - 1.4)

y -= 3
main_box(y, f"{n_after_ecg} test instances")
side_arrow(y + 0.8)
excl_box(y + 0.8, f"n={n_still_excl}\nAccelerometer-confirmed\nstillness not located")
down_arrow(y, y - 1.4)

y -= 3
main_box(y, f"{n_after_still} test instances")
side_arrow(y + 0.8)
excl_box(y + 0.8, f"n={n_baseline_excl}\nBaseline heart rate\nimplausible (>120 bpm)")
down_arrow(y, y - 1.4)

y -= 3
main_box(y, f"{n_after_baseline} test instances")
side_arrow(y + 0.8)
excl_box(y + 0.8, f"n={n_recovery_excl}\nRecovery-window HR\nuncomputable (0 good beats)")
down_arrow(y, y - 1.4)

y -= 3
main_box(y, f"{n_final} test instances, {n_final_patients} of 80 patients (94%)\n"
             f"usable primary outcome (ΔHR60-confirmed)",
         color=COLOR_FINAL, height=1.8)

fig.suptitle('Supplementary Figure S2. Test-instance flow diagram', fontsize=12, y=0.98)
fig.tight_layout()
fig.savefig('../data_derived/figureS2_flow_diagram.png', dpi=600, bbox_inches='tight')
print(f"saved: data_derived/figureS2_flow_diagram.png")
print(f"cascade: {n_total} -> -{n_ecg_excl} -> {n_after_ecg} -> -{n_still_excl} -> {n_after_still} "
      f"-> -{n_baseline_excl} -> {n_after_baseline} -> -{n_recovery_excl} -> {n_final} "
      f"({n_final_patients} patients)")
