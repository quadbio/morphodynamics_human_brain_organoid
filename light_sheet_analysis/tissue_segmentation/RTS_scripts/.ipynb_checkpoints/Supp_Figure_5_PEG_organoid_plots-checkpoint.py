"""Script for generating PEG plots Supp figure 5 f and g"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Get file path for each conditions
PEG_RDG_MINUS_CTRL = ""
PEG_RDG_PLUS_LAMININ = ""
PEG_GFO_MINUS = ""

# Define output folder for plots
OUTPUT_FLD = ""

custom_palette = {
    "RDG_minus_ctrl": "#353440",
    "RDG_plus_laminin": "#0554F2",
    "GFO_minus": "#026E81",
}

# Read in files
# Assign labels to distinguish the different conditions
res_df_RDG_minus_ctrl = pd.read_csv(PEG_RDG_MINUS_CTRL, sep=",")
res_df_RDG_minus_ctrl["Sample"] = "RDG_minus_ctrl"
res_df_RDG_minus_ctrl["experiment"] = "PEG_exp"

res_df_RDG_plus_laminin = pd.read_csv(PEG_RDG_PLUS_LAMININ, sep=",")
res_df_RDG_plus_laminin["Sample"] = "RDG_plus_laminin"
res_df_RDG_plus_laminin["experiment"] = "PEG_exp"

res_df_GFO_minus = pd.read_csv(PEG_GFO_MINUS, sep=",")
res_df_GFO_minus["Sample"] = "PEG_GFO_minus"
res_df_GFO_minus["experiment"] = "PEG_exp"

# Concatenate all conditions into one data frame
df_all_conditions = res_df_RDG_minus_ctrl
df_all_conditions = pd.concat(
    [df_all_conditions, res_df_RDG_plus_laminin], ignore_index=True
)
df_all_conditions = pd.concat([df_all_conditions, res_df_GFO_minus], ignore_index=True)

# Calculate time
time_in_days = ((df_all_conditions["timepoint"] - 1) / 24) + 4
df_all_conditions["time_in_days"] = time_in_days
df_all_conditions["ratio_lumen_o_organoid"] = (
    df_all_conditions["lumen_volume"] / df_all_conditions["organoid_volume"]
)

# (lumen volume)/(lumen+epithelium volume)
df_all_conditions["percent_lumen_vol"] = df_all_conditions["lumen_volume"] / (
    df_all_conditions["lumen_volume"] + df_all_conditions["organoid_volume"]
)

# Plot number of lumen and lumen volume
sns.lineplot(
    df_all_conditions,
    x="time_in_days",
    y="number_of_lumen",
    hue="Sample",
    errorbar="sd",
    palette=custom_palette.values(),
)
plt.xlabel("time[days]")
plt.ylabel("number_of_lumen")
plt.title("PEG number of lumen")
plt.savefig(OUTPUT_FLD + "PEG_number_of_lumen.pdf")
plt.show()
plt.close()

sns.lineplot(
    df_all_conditions,
    x="time_in_days",
    y="lumen_volume",
    hue="Sample",
    errorbar="sd",
    palette=custom_palette.values(),
)
plt.xlabel("time[days]")
plt.ylabel("lumen_volume")
plt.title("PEG lumen volume")
plt.savefig(OUTPUT_FLD + "PEG_lumen_volume.pdf")
plt.show()
plt.close()
