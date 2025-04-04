"""Script to extract process and plot agarose 06 and 03 percent data"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Create custom colour palette
palette_custom = {"Agarose_06": "#374484", "Agarose_03": "#69BCDE"}


# File path to 0.3 percent agarose data (csv)
FILE_PATH_AGAR_03 = ""

# Get dir from preprint figure 0.6 percent agarose data (csv)
FILE_PATH_AGAR_06_LUMEN_VOLUME = ""
FILE_PATH_AGAR_06_NUMBER_OF_LUMEN = ""
FILE_PATH_AGAR_06_ORGANOID_VOLUME = ""

# Set output folder for plots
OUTPUT_FLD = ""

# Read files agarose 0.3 percent
# Assign labels to distinguish the different dataframes
df_agar_03 = pd.read_csv(FILE_PATH_AGAR_03, sep=",")
df_agar_03["type"] = "Agarose 0.3%"

# Read files agarose 0.6 percent
lumen_volume_df_agar_06 = pd.read_csv(FILE_PATH_AGAR_06_LUMEN_VOLUME)
n_lumen_df_agar_06 = pd.read_csv(FILE_PATH_AGAR_06_NUMBER_OF_LUMEN)
organoid_volume_df_agar_06 = pd.read_csv(FILE_PATH_AGAR_06_ORGANOID_VOLUME)

# Prepare dataframe, make it long format, extract conditions and
# organoid number from the sample name
# Make the df long format, eg column form for Organoids


def to_long_form(input_data, timepoints, condition, modality):
    """Converts wide format pandas df into long format extract condition name
    and timepoint from sample name"""

    df_converted = input_data.melt(id_vars=timepoints, value_vars=condition)
    df_converted["image_number"] = df_converted["variable"].str.extract(r"(\d+)")
    df_converted["condition"] = df_converted["variable"].str.extract(
        "(matrigel|no ecm|in agar)"
    )
    df_converted = df_converted.rename(
        columns={timepoints: "timepoint", "value": modality}
    )

    return df_converted


organoid_samples = [
    "organoid (matrigel) 1",
    "organoid (matrigel) 2",
    "organoid (matrigel) 3",
    "organoid (matrigel) 4",
    "organoid (no ecm) 5",
    "organoid (no ecm) 6",
    "organoid (no ecm) 7",
    "organoid (no ecm) 8",
    "organoid (in agar) 9",
    "organoid (in agar) 10",
    "organoid (in agar) 11",
    "organoid (in agar) 12",
    "organoid (no ecm) 13",
    "organoid (no ecm) 14",
    "organoid (no ecm) 15",
    "organoid (no ecm) 16",
]

df_agar_lumen_volume = to_long_form(
    lumen_volume_df_agar_06,
    timepoints="Unnamed: 0",
    condition=organoid_samples,
    modality="lumen_volume",
)

df_agar_number_of_lumen = to_long_form(
    n_lumen_df_agar_06,
    timepoints="Unnamed: 0",
    condition=organoid_samples,
    modality="number_of_lumen",
)

df_agar_organoid_volume = to_long_form(
    organoid_volume_df_agar_06,
    timepoints="Unnamed: 0",
    condition=organoid_samples,
    modality="organoid_volume",
)

# Combine all agar dataframes into one
df_all_modalities = df_agar_lumen_volume.join(
    df_agar_number_of_lumen["number_of_lumen"]
)
df_all_modalities = df_all_modalities.join(df_agar_organoid_volume["organoid_volume"])

ag06 = df_all_modalities[df_all_modalities["condition"] == "in agar"]
ag06["type"] = "Agarose 0.6%"


# List all timepoints used in the agarose 03 organoids
timepoints_ag03 = list(df_agar_03["timepoint"])

# Extract all timepoints that are common between ag03 and ag06
mask = ag06["timepoint"].isin(timepoints_ag03)

# Downsample agarose 06 data with timepoints used in the agarose 03 samples
ds_ag06 = ag06[mask]


# Concatenate agarose 03 and downsampled agarose 06 datasets
df_concat_ds = pd.concat([df_agar_03, ds_ag06], ignore_index=True)

# Calculate time in days and add as a new column
df_concat_ds["time_in_days"] = ((df_concat_ds["timepoint"] - 1) / 24) + 4
df_concat_ds["ratio_lumen_o_organoid"] = (
    df_concat_ds["lumen_volume"] / df_concat_ds["organoid_volume"]
)

# Calculate percent lumen volume: lumen volume/(lumen+epithelium volume)
df_concat_ds["percent_lumen_vol"] = df_concat_ds["lumen_volume"] / (
    df_concat_ds["lumen_volume"] + df_concat_ds["organoid_volume"]
)

# Calculate lumen volume/organoid volume
df_concat_ds["lumen_per_org_volume"] = (
    df_concat_ds["number_of_lumen"] / df_concat_ds["organoid_volume"]
)

# Plot agarose 06 vs 03 percent data. AJ recommended plots only go up to day 7
# This is because after Day 7 the organoids move a lot in the field of view
df_agarose_up_to_d7 = df_concat_ds[df_concat_ds["time_in_days"] <= 7]

# Plot the percent lumen volume
sns.lineplot(
    df_agarose_up_to_d7,
    x="time_in_days",
    y="percent_lumen_vol",
    hue="type",
    errorbar="sd",
    palette=palette_custom.values(),
)
plt.xlabel("time[days]")
plt.ylabel("lumen volume / (lumen + organoid volume)")
plt.title("Percent lumen volume")
plt.savefig(OUTPUT_FLD + "agar_per_lumen_vol_ds.pdf")
plt.close()

# Plot number of lumen over organoid volume
sns.lineplot(
    df_agarose_up_to_d7,
    x="time_in_days",
    y="lumen_per_org_volume",
    hue="type",
    errorbar="sd",
    palette=palette_custom.values(),
)
plt.xlabel("time[days]")
plt.ylabel("Number of segmented lumen / organoid volume")
plt.title("Number of segmented lumen over organoid volume")
plt.savefig(OUTPUT_FLD + "agar_lumen_over_organoid_ds.pdf")
plt.close()
