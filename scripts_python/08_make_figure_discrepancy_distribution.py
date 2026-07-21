"""
Figure: distribution of the naive-to-confirmed rest-onset discrepancy across
all instances, split by test type, with the motion-verification result
(movement energy at naive onset vs. patient-specific quiet threshold) shown
alongside it. This is the headline-finding figure: it lets a reader see, at
a glance and across the whole sample rather than one anecdote (Figure 2),
that the gap between naive and confirmed timing is (a) large and (b) driven
by genuine ongoing movement in the great majority of instances.

Panel A: histogram of discrepancy_sec (all 164 instances with a computable
naive-vs-confirmed comparison), colored by test type.
Panel B: for the 149 instances with a re-derivable accelerometer trace,
the fraction whose movement energy at the naive-onset timestamp exceeded
the patient's own quiet threshold, by test type, with the overall 90.6%
figure marked as a reference line.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEST_COLORS = {"TUG": "#2a78d6", "VELO": "#eb6834", "6MWT": "#1baf7a"}
TEST_ORDER = ["TUG", "VELO", "6MWT"]

hrr = pd.read_csv('../data_derived/hrr_features_all.csv')
disc = hrr[hrr['discrepancy_sec'].notna()].copy()
print(f"n instances with valid discrepancy: {len(disc)}")

verif = pd.read_csv('../data_derived/discrepancy_motion_verification.csv')
print(f"n instances with motion verification: {len(verif)}")

overall_pct = 100 * verif['naive_end_above_threshold'].mean()
by_test_pct = verif.groupby('test_type')['naive_end_above_threshold'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
bins = [x for x in range(-60, 400, 20)]
for tt in TEST_ORDER:
    vals = disc.loc[disc['test_type'] == tt, 'discrepancy_sec']
    ax.hist(vals, bins=bins, alpha=0.55, label=f"{tt} (n={len(vals)})",
            color=TEST_COLORS[tt], edgecolor='white', linewidth=0.4)
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.set_xlabel("Naive-to-confirmed rest-onset discrepancy (s)")
ax.set_ylabel("Number of test instances")
ax.set_title("A. Distribution of timing discrepancy")
ax.legend(fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
bar_vals = [by_test_pct.get(tt, 0) for tt in TEST_ORDER]
bar_n = [int((verif['test_type'] == tt).sum()) for tt in TEST_ORDER]
bars = ax.bar(TEST_ORDER, bar_vals, color=[TEST_COLORS[tt] for tt in TEST_ORDER],
              edgecolor='white', linewidth=0.6)
for b, n in zip(bars, bar_n):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"n={n}",
            ha='center', va='bottom', fontsize=8)
ax.axhline(overall_pct, color='black', linewidth=1.2, linestyle='--')
ax.text(2.45, overall_pct + 2, f"overall: {overall_pct:.1f}%", ha='right',
        va='bottom', fontsize=8.5)
ax.set_ylim(0, 108)
ax.set_ylabel("Instances still moving at naive-onset\ntimestamp (%)")
ax.set_title("B. Motion verification at naive onset")
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle("Naive-to-confirmed timing discrepancy is large and reflects genuine motion",
             fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig('../data_derived/fig2_discrepancy_distribution.png', dpi=600, bbox_inches='tight')
print("saved: data_derived/fig2_discrepancy_distribution.png")
