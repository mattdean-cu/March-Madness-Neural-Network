"""
07_combined_figure.py
=====================
Generates a single combined figure showing all 6 rounds of the
2026 NCAA Tournament predictions stacked vertically.
Save to figures/2026_all_rounds.png

Run AFTER 06_predict_2026.py has generated the individual figures.

Author: [Your Name]
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FIGURES_DIR = "./figures"

round_files = [
    ("Round 1 (32 games)",       "2026_tournament_predictions.png"),
    ("Round of 32 (16 games)",   "2026_r2_predictions.png"),
    ("Sweet 16 (8 games)",       "2026_s16_predictions.png"),
    ("Elite Eight (4 games)",    "2026_e8_predictions.png"),
    ("Final Four (2 games)",     "2026_ff_predictions.png"),
    ("Championship (1 game)",    "2026_championship_prediction.png"),
]

fig, axes = plt.subplots(len(round_files), 1,
                          figsize=(18, 6 * len(round_files)))

fig.suptitle(
    "2026 NCAA Tournament — Neural Network Predictions vs Actual Results\n"
    "Winner shown on left  |  Bar length = model confidence  |  ★ = actual upset  |  Champion: Michigan",
    fontsize=14, fontweight="bold", y=1.005
)

for ax, (label, fname) in zip(axes, round_files):
    path = os.path.join(FIGURES_DIR, fname)
    if not os.path.exists(path):
        ax.text(0.5, 0.5, f"Missing: {fname}", ha="center", va="center",
                fontsize=12, color="red")
        ax.axis("off")
        continue

    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(label, fontsize=12, fontweight="bold",
                 loc="left", pad=6, color="#1e3a5f")

plt.tight_layout(h_pad=0.5)
out = os.path.join(FIGURES_DIR, "2026_all_rounds.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"[Saved] {out}")
plt.show()