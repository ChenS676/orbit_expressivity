import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

# # === Define full raw_data with α = 0.6 ===
# """
# max_orbit_gcn_alchemy_2_orbit_sorting_cross_entropy_10_100,0.581+- 0.006, 0.5845+- 0.013, 0.000 +- 0.000
# max_orbit_gcn_alchemy_3_orbit_sorting_cross_entropy_10_100,0.551 +- 0.008,0.538 +- 0.013,0.000 +- 0.000
# max_orbit_gcn_alchemy_4_orbit_sorting_cross_entropy_10_100,0.541 +- 0.016,0.531 +- 0.023,0.000 +- 0.000
# max_orbit_gcn_alchemy_5_orbit_sorting_cross_entropy_10_100,0.541 +- 0.014,0.525 +- 0.023,0.000 +- 0.000
# max_orbit_gcn_alchemy_6_orbit_sorting_cross_entropy_10_100,0.533 +- 0.018,0.524 +- 0.037,0.000 +- 0.000
# max_orbit_gcn_alchemy_9_orbit_sorting_cross_entropy_10_100,0.452 +- 0.023,0.456 +- 0.035,0.000 +- 0.000
# max_orbit_gcn_alchemy_9_orbit_sorting_cross_entropy_3_100,0.439 +- 0.014,0.441 +- 0.020,0.000 +- 0.000
# max_orbit_gcn_alchemy_10_orbit_sorting_cross_entropy_10_100,0.435 +- 0.008,0.435 +- 0.013,0.000 +- 0.000



# gcn_alchemy_10_orbit_sorting_cross_entropy_3_100,0.499 +- 0.005,0.681 +- 0.001,0.000 +- 0.000
# gcn_alchemy_9_orbit_sorting_cross_entropy_3_100,0.474 +- 0.001,0.682 +- 0.000,0.000 +- 0.000
# gcn_alchemy_9_orbit_sorting_cross_entropy_3_100,0.497 +- 0.004,0.681 +- 0.001,0.000 +- 0.000
# gcn_alchemy_7_orbit_sorting_cross_entropy_3_100,0.498 +- 0.005,0.681 +- 0.001,0.000 +- 0.000
# gcn_alchemy_6_orbit_sorting_cross_entropy_3_100,0.501 +- 0.005,0.681 +- 0.001,0.000 +- 0.000
# gcn_alchemy_5_orbit_sorting_cross_entropy_3_100,0.496 +- 0.004,0.681 +- 0.001,0.000 +- 0.000
# gcn_alchemy_4_orbit_sorting_cross_entropy_3_100,0.500 +- 0.002,0.680 +- 0.002,0.000 +- 0.000
# gcn_alchemy_3_orbit_sorting_cross_entropy_3_100,0.496 +- 0.003,0.681 +- 0.001,0.000 +- 0.000
# gcn_alchemy_2_orbit_sorting_cross_entropy_3_100,0.500 +- 0.004,0.681 +- 0.001,0.000 +- 0.000



# === Structured raw_data ===
node_accuracy = [
    ("MaxOrbitGCN",
     [2, 3, 4, 5, 6, 9, 9, 10],
     [0.581, 0.551, 0.541, 0.533, 0.452, 0.439, 0.435, 0.430],
     [0.006, 0.008, 0.016, 0.014, 0.018, 0.023, 0.014, 0.008]),

    ("GCN",
     [2, 3, 4, 5, 6, 7, 8, 9, 10],
     [0.499, 0.474, 0.497, 0.501, 0.496, 0.500, 0.496, 0.500, 0.500],
     [0.005, 0.001, 0.004, 0.005, 0.004, 0.002, 0.003, 0.004, 0.004]),
]

orbit_accuracy = [
    ("MaxOrbitGCN",
     [2, 3, 4, 5, 6, 7, 9, 9, 10],
     [0.5845, 0.538, 0.531, 0.525, 0.524, 0.524, 0.456, 0.441, 0.435],
     [0.013, 0.013, 0.023, 0.023, 0.037, 0.037, 0.035, 0.020, 0.013]), 
    
    ("GCN",
     [2, 3, 4, 5, 6, 7, 9, 9, 10],
     [0.681, 0.682, 0.681, 0.681, 0.680, 0.681, 0.681, 0.681, 0.681],
     [0.001, 0.000, 0.001, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001]),
]
raw_data = orbit_accuracy
# === Prepare plot data ===
plot_data = defaultdict(dict)
for model, alpha, best_valid, variance in raw_data:
    plot_data[model]["alpha"] = alpha
    plot_data[model]["best_valid"] = best_valid
    plot_data[model]["variance"] = variance

baselines = ["GCN", "MaxOrbitGCN"]

# === Color setup ===
def is_yellow(rgb): return rgb[0] > 0.9 and rgb[1] > 0.9 and rgb[2] < 0.6
palette = sns.color_palette("Set2", len(baselines) + 2)
baseline_colors = [c for c in palette if not is_yellow(c)][:len(baselines)]
model_colors = {m: c for m, c in zip(baselines, baseline_colors)}

dashed_models = {"ChebGCN", "LINKX", "GIN"}
line_styles = {m: "--" if m in dashed_models else "-" for m in plot_data}

# === Plotting ===
fig, ax = plt.subplots(figsize=(10, 8))

for idx, (model, values) in enumerate(plot_data.items()):
    color = model_colors.get(model, f"C{idx}")
    alphas = values["alpha"]
    scores = values["best_valid"]
    variances = values["variance"]

    ax.plot(
        alphas, scores,
        linestyle=line_styles.get(model, "-"),
        linewidth=2.2, color=color,
        label=model, marker='o',
        markersize=5.5, markerfacecolor=color,
        markeredgecolor='black', markeredgewidth=0.6
    )

    lower_var = np.array(variances) * 0.8
    upper_var = np.array(variances) * 0.2

    ax.errorbar(
        alphas, scores, yerr=[lower_var, upper_var],
        fmt='o', color=color, alpha=0.25,
        capsize=4, elinewidth=1.4, capthick=1.4
    )

# === Axis labels and ticks ===
fontsize = 16
ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
ax.set_ylabel("ACC (/%)", fontsize=fontsize)

# Ensure all unique alphas across models
all_alphas = sorted(set(sum([v["alpha"] for v in plot_data.values()], [])))
ax.set_xticks(all_alphas)

# Uniform y-axis from 40 to 60
ax.set_ylim(0.40, 0.70)
ax.set_yticks(np.arange(0.40, 0.70, 0.04))

ax.tick_params(axis='both', labelsize=fontsize)
ax.legend(fontsize=13, loc="lower left", frameon=False, ncol=1)

plt.tight_layout()
plt.savefig("Orbit_Acc.pdf", bbox_inches='tight')
plt.show()