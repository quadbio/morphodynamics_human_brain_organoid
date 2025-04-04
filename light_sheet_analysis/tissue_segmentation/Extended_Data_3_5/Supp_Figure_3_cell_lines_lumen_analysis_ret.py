"""Script for analysis of 100 percent labeled cell lines"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.io
from scipy.stats import f_oneway, mannwhitneyu
from skimage import exposure
from skimage.io import imread
from statsmodels.stats.multitest import multipletests

# Set figure context and format
sns.set_context("paper", rc={"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8})
rc = {
    "figure.figsize": (5, 4),
    "axes.facecolor": "white",
    "axes.grid": False,
    "grid.color": ".8",
    "font.family": "Arial",
    "font.size": 5,
}
plt.rcParams.update(rc)
CM = 1 / 2.54  # centimeters in inches
fig, ax = plt.subplots(figsize=(4 * CM, 3.2 * CM))
sns.despine(left=True, bottom=True, right=True)
pallette = {"Matrigel": "#17ad97", "No Matrix": "#4d4d4d", "Agarose": "#98d9d1"}


# Create an empty data frame
# Read in masks and loop over them
# Image acquisition done by AJ, some dont have the correct label
# E.g. some files named laminin are tubulin. Images were checked
# For correct cell labels

# Define input folder path for images
INPUT_FLD = ""

# Input folder path for lumen masks
MASK_FLD = ""

# Define output folder path
OUTPUT_FLD = ""

# Tubulin (Images: 5, 7, 8)

T1 = INPUT_FLD + "tubulin_all_m05to07/p5_t/"
T2 = INPUT_FLD + "tubulin_all_m05to07/p7_t/"
T3 = INPUT_FLD + "laminin_all_m08to10/p8_l/"

# Lamin (Images: 6, 9, 12)
L1 = INPUT_FLD + "tubulin_all_m05to07/p6_t/"
L2 = INPUT_FLD + "laminin_all_m08to10/p9_l/"
L3 = INPUT_FLD + "actin_all_m11to13/p12_ac/"

# Actin (Images: 10, 11, 13)
A1 = INPUT_FLD + "laminin_all_m08to10/p10_l/"
A2 = INPUT_FLD + "actin_all_m11to13/p11_ac/"
A3 = INPUT_FLD + "actin_all_m11to13/p13_ac/"

# Histone (Images: 14, 15, 16)
H1 = INPUT_FLD + "histone_all_m14to16/p14_h/"
H2 = INPUT_FLD + "histone_all_m14to16/p15_h/"
H3 = INPUT_FLD + "histone_all_m14to16/p16_h/"

# Add input dir variables to list, and also create list with all replicate
# names, and condition names
input_dirs = [T1, T2, T3, L1, L2, L3, A1, A2, A3, H1, H2, H3]
str_input_dirs = [
    "T1",
    "T2",
    "T3",
    "L1",
    "L2",
    "L3",
    "A1",
    "A2",
    "A3",
    "H1",
    "H2",
    "H3",
]
cond_list = [
    "Tubulin",
    "Tubulin",
    "Tubulin",
    "Lamin",
    "Lamin",
    "Lamin",
    "Actin",
    "Actin",
    "Actin",
    "Histone",
    "Histone",
    "Histone",
]

# Image mask collage date read in
image_path_list = [
    MASK_FLD + "/t0121_561_P5.tif",
    MASK_FLD + "/t0121_561_P6.tif",
    MASK_FLD + "/t0121_488_P10.tif",
    MASK_FLD + "/t0121_488_P15.tif",
]

mask_path_list = [
    (INPUT_FLD + "tubulin_all_m05to07/p5_t/" "0121_lumen_organoid_mask_processed.tif"),
    (INPUT_FLD + "tubulin_all_m05to07/p6_t/" "0121_lumen_organoid_mask_processed.tif"),
    (INPUT_FLD + "laminin_all_m08to10/p10_l/" "0121_lumen_organoid_mask_processed.tif"),
    (INPUT_FLD + "histone_all_m14to16/p15_h/" "0121_lumen_organoid_mask_processed.tif"),
]

# Create data frame with all
# Create an empty data frame for the lumen analysis
lumen_analysis = pd.DataFrame()

# Counter to 0 (Used to cycle through the condition and replicate lists)
COUNTER = 0

# For loop over all stacks and timepoints, fills out lumen analysis dataframe
for inp_dir in input_dirs:
    for stack in ["0001", "0025", "0049", "0073", "0097", "0121"]:
        # Load masks from tif
        combined_masks = imread(
            inp_dir + f"{stack:04}" + "_lumen_organoid_mask_processed.tif"
        )
        lumen_mask = combined_masks == 3

        if "H" in str_input_dirs[COUNTER] or "A" in str_input_dirs[COUNTER]:
            upscaled_img = skimage.transform.rescale(
                lumen_mask, [0.5 / (4 * 0.347), 1, 1], order=0, preserve_range=True
            )

        else:
            upscaled_img = skimage.transform.rescale(
                lumen_mask, [2 / (4 * 0.347), 1, 1], order=0, preserve_range=True
            )

        cleaned_mask = skimage.morphology.remove_small_objects(
            upscaled_img, min_size=20000 / ((0.347 * 4) ** 3)
        )
        label_img, num_labels = skimage.measure.label(cleaned_mask, return_num=True)
        regions_table = skimage.measure.regionprops_table(
            label_img, properties=["label", "area", "axis_major_length"]
        )

        tp_df = pd.DataFrame.from_dict(regions_table)

        tp_df["timepoint"] = ((int(stack) - 1) / 24) + 4
        tp_df["replicate"] = str_input_dirs[COUNTER]
        tp_df["condition"] = cond_list[COUNTER]

        lumen_analysis = pd.concat([lumen_analysis, tp_df], ignore_index=True)

    COUNTER += 1

# Convert to µm^3 for area (Volume)
lumen_analysis["area_converted"] = lumen_analysis["area"] * (4 * 0.347) ** 3

# Convert to µm for axis
lumen_analysis["axis_major_length_converted"] = lumen_analysis["axis_major_length"] * (
    0.347 * 4
)

# Subset to day 9 for volume (area) and axis major length
# (wilcoxon rank sum test)
tub9_area = lumen_analysis[
    (lumen_analysis["timepoint"] == 9) & (lumen_analysis["condition"] == "Tubulin")
]
lam9_area = lumen_analysis[
    (lumen_analysis["timepoint"] == 9) & (lumen_analysis["condition"] == "Lamin")
]
act9_area = lumen_analysis[
    (lumen_analysis["timepoint"] == 9) & (lumen_analysis["condition"] == "Actin")
]
his9_area = lumen_analysis[
    (lumen_analysis["timepoint"] == 9) & (lumen_analysis["condition"] == "Histone")
]

tub9_axisml = lumen_analysis[
    (lumen_analysis["timepoint"] == 9) & (lumen_analysis["condition"] == "Tubulin")
]
lam9_axisml = lumen_analysis[
    (lumen_analysis["timepoint"] == 9) & (lumen_analysis["condition"] == "Lamin")
]
act9_axisml = lumen_analysis[
    (lumen_analysis["timepoint"] == 9) & (lumen_analysis["condition"] == "Actin")
]
his9_axisml = lumen_analysis[
    (lumen_analysis["timepoint"] == 9) & (lumen_analysis["condition"] == "Histone")
]

# Statistical tests
# ANOVA
# Assumptions:

measurements = ["area_converted", "axis_major_length_converted"]
conditions = ["Tubulin", "Lamin", "Actin", "Histone"]
results = pd.DataFrame()

for anova_measurement in measurements:
    for day in range(4, 10):
        # Collect the area_converted data for each condition at the given day
        data_for_day = [
            lumen_analysis[
                (lumen_analysis["condition"] == cond)
                & (lumen_analysis["timepoint"] == day)
            ][anova_measurement]
            for cond in conditions
        ]

        anova_results = f_oneway(*data_for_day)
        # Run the one-way ANOVA for that day and store the result
        results_one_line = pd.DataFrame(
            [
                [
                    f"day{day}_ooanova_{anova_measurement}",
                    anova_results[1],
                    anova_results[0],
                ]
            ],
            columns=["comparison", "pval_anova", "F_statistic"],
            index=[f"Day_{day}_{anova_measurement}"],
        )

        results = pd.concat([results, results_one_line])

# False discovery rate

results_fdr_area = multipletests(results["pval_anova"].iloc[:6], method="bonferroni")
results_fdr_axis = multipletests(results["pval_anova"].iloc[6:], method="bonferroni")

results["padj_anova"] = np.hstack((results_fdr_area[1], results_fdr_axis[1]))

results.to_csv(OUTPUT_FLD + "anova_stats_table.csv")

# Wilcoxon rank sum test day 9
# Assumptions:
# Volume (area)

measurements = ["area_converted", "axis_major_length_converted"]
conditions = ["Tubulin", "Lamin", "Actin", "Histone"]

result_wilcoxon_df = pd.DataFrame()

lumen_analysis_day9 = lumen_analysis[lumen_analysis["timepoint"] == 9]

for measurement in measurements:
    results_wilcoxon_measurement = pd.DataFrame()
    for condition_1 in conditions:
        for condition_2 in conditions:
            if condition_1 != condition_2:
                wilcoxon_test = mannwhitneyu(
                    lumen_analysis_day9[
                        lumen_analysis_day9["condition"] == condition_1
                    ][measurement],
                    lumen_analysis_day9[
                        lumen_analysis_day9["condition"] == condition_2
                    ][measurement],
                )

                results_one_line_wilcoxon = pd.DataFrame(
                    [
                        [
                            f"day_9_wilcoxon_{measurement}_{condition_1}_{condition_2}",
                            wilcoxon_test[1],
                            wilcoxon_test[0],
                        ]
                    ],
                    columns=["comparison", "pval_wilcoxon_rank_sum", "U_statistic"],
                    index=[f"Day_9_{measurement}"],
                )

                results_wilcoxon_measurement = pd.concat(
                    [results_wilcoxon_measurement, results_one_line_wilcoxon]
                )

    results_wilcoxon_measurement["padj_wilcoxon_rank_sum"] = multipletests(
        results_wilcoxon_measurement["pval_wilcoxon_rank_sum"], method="bonferroni"
    )[1]

    result_wilcoxon_df = pd.concat([result_wilcoxon_df, results_wilcoxon_measurement])

result_wilcoxon_df.to_csv("/Users/rtschannen/Desktop/wilcoxon_stats_table.csv")

measurements = ["area_converted", "axis_major_length_converted"]
conditions = ["Tubulin", "Lamin", "Actin", "Histone"]

result_wilcoxon_df = pd.DataFrame()

lumen_analysis_day9 = lumen_analysis[lumen_analysis["timepoint"] == 9]

for measurement in measurements:
    results_wilcoxon_measurement = pd.DataFrame()
    # Use indices to ensure each pair is only tested once
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            condition_1 = conditions[i]
            condition_2 = conditions[j]

            data_1 = lumen_analysis_day9[
                lumen_analysis_day9["condition"] == condition_1
            ][measurement]
            data_2 = lumen_analysis_day9[
                lumen_analysis_day9["condition"] == condition_2
            ][measurement]

            wilcoxon_test = mannwhitneyu(data_1, data_2, alternative="two-sided")

            results_one_line_wilcoxon = pd.DataFrame(
                [
                    [
                        f"day_9_wilcoxon_{measurement}_{condition_1}_{condition_2}",
                        wilcoxon_test.pvalue,
                        wilcoxon_test.statistic,
                    ]
                ],
                columns=["comparison", "pval_wilcoxon_rank_sum", "U_statistic"],
                index=[f"Day_9_{measurement}"],
            )

            results_wilcoxon_measurement = pd.concat(
                [results_wilcoxon_measurement, results_one_line_wilcoxon]
            )

    # Adjust p-values
    results_wilcoxon_measurement["padj_wilcoxon_rank_sum"] = multipletests(
        results_wilcoxon_measurement["pval_wilcoxon_rank_sum"], method="bonferroni"
    )[1]

    result_wilcoxon_df = pd.concat([result_wilcoxon_df, results_wilcoxon_measurement])

result_wilcoxon_df.to_csv("/Users/rtschannen/Desktop/wilcoxon_stats_table.csv")

# Violin plot 1
sns.violinplot(
    lumen_analysis,
    x="timepoint",
    y="area_converted",
    hue="condition",
    log_scale=True,
    cut=0,
)

plt.title("Lumen analysis: Violinplot linevolume over time", fontsize=10)
plt.legend(loc="lower right")
plt.ylabel(r"Volume $[\mu m^3]$")
plt.savefig(OUTPUT_FLD + "linevolumeot.pdf")
plt.close()

# Violin plot 2
sns.violinplot(
    lumen_analysis,
    x="timepoint",
    y="axis_major_length_converted",
    hue="condition",
    log_scale=False,
    cut=0,
)

plt.title(
    "Lumen analysis: Violinplot fit ellipse major axis length over time", fontsize=10
)
plt.legend(loc="center right")
plt.ylabel(r"Axis major length $[\mu m]$")
plt.savefig(OUTPUT_FLD + "axismajorot.pdf")
plt.close()

# Violin plot 3 conditions volume (area)
sns.violinplot(
    lumen_analysis[lumen_analysis["timepoint"] == 9],
    x="timepoint",
    y="area_converted",
    hue="condition",
    log_scale=True,
    cut=0,
)
plt.title("Lumen analysis: Violinplot linevolume Day 9 for all conditions", fontsize=10)
plt.legend(loc="upper right")
plt.ylabel(r"Volume $[\mu m^3]$")
plt.xticks([])
plt.savefig(OUTPUT_FLD + "day9conditionsvolumeviolinplot.pdf")
plt.close()


# Violin plot 3 conditions axis major length
sns.violinplot(
    lumen_analysis[lumen_analysis["timepoint"] == 9],
    x="timepoint",
    y="axis_major_length_converted",
    hue="condition",
    log_scale=False,
    cut=0,
)
plt.title(
    "Lumen analysis: Violinplot axis major length Day 9 for all conditions", fontsize=10
)
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
plt.ylabel(r"Axis major length $[\mu m]$")
plt.xticks([])
plt.savefig(OUTPUT_FLD + "day9conditionsaxismajorlenviolinplot.pdf")
plt.close()

# Imaging mask creation

montage_list = []
channel_list = [True, True, False, False]

for image_path, mask_path, channel in zip(
    image_path_list, mask_path_list, channel_list
):
    mask = imread(mask_path)
    image = imread(image_path)

    if channel:
        mask = skimage.morphology.remove_small_objects(
            mask, min_size=20000 / ((0.347 * 4 * 0.347 * 4 * 2))
        )
    else:
        mask = skimage.morphology.remove_small_objects(
            mask, min_size=20000 / ((0.347 * 4 * 0.347 * 4 * 0.5))
        )

    img_number = image[image.shape[0] // 4]
    mask_number = mask[mask.shape[0] // 4]

    img_number = exposure.rescale_intensity(
        img_number, in_range=(0, np.percentile(img_number, 99)), out_range=(0, 1)
    )

    assert image.shape[0] == mask.shape[0]

    upscaled_mask = skimage.transform.rescale(
        mask_number, [4, 4], order=0, preserve_range=True
    )

    stack_sep_channel = []

    for masked_ind in range(2, 4):
        stack_sep_channel.append((upscaled_mask == masked_ind) * img_number)

    stack_sep_channel = np.array(stack_sep_channel)

    montage_list.append(stack_sep_channel)

montage_array = np.dstack(np.array(montage_list))

dpi = mpl.rcParams["figure.dpi"]
fig = plt.figure(
    figsize=(montage_array[0].shape[1] / dpi, montage_array[0].shape[0] / dpi)
)
fig.tight_layout()
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(montage_array[0], cmap="grey")
ax.imshow(montage_array[1], cmap="inferno", alpha=0.50)

ax.axis("off")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.savefig(
    OUTPUT_FLD + "montage_cell_lines_fully_labelled.png",
    pad_inches=0,
    bbox_inches="tight",
    dpi=dpi,
)
plt.close()
