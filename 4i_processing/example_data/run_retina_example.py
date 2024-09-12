# Example script to process 4i data with phenoscapes
# This script downloads example 4i data of retinal organoids from Wahle et al. (2023) and processes it with phenoscapes
# 'Multimodal spatiotemporal phenotyping of human retinal organoid development', Nature Biotechnology volume 41, pages 1765–1775 (2023)

import os
from pathlib import Path
import numpy as np
from phenoscapes.utils import get_metadata
from phenoscapes.cli import process_sample, load_default_config
import yaml

# make directory for example data
os.makedirs('4i_retina_example', exist_ok=True)
# set working directory
os.chdir('4i_retina_example')
dir_4i_retina_example = os.getcwd()

# if example data not downloaded yet
if not os.path.exists('stitched'):
    if not os.path.exists('example_data.tar'):
        # download example data
        os.system('wget https://zenodo.org/record/7561908/files/example_data.tar?download=1 -O example_data.tar')
    # extract example data
    os.system('tar -xvf example_data.tar')
    # rename example data to stitched
    os.system('mv example_data stitched')

# list files in stitched
files = os.listdir('stitched')
# rename files if 'excluded ' in file name
for file in files:
    if 'excluded ' in file:
        new_name = file.replace('excluded ', '')
        os.rename(os.path.join('stitched', file), os.path.join('stitched', new_name))


# from file names generate dataframe with sample, stain, cycle_id, channel_id
regex_retina = r'cycle(?P<cycle_id>\d{1,2})_well(?P<well_id>\d{2})_channel(?P<channel_id>\d{1})_(?P<stain>.+)\.tif$'

df = get_metadata('stitched', regex_retina)

# generate stain metadata from file names
df_stains = df[['cycle_id', 'channel_id', 'stain']].drop_duplicates().sort_values(by=['cycle_id', 'channel_id'])
# remove 'excluded ' from stain names
df_stains['stain'] = df_stains['stain'].apply(lambda x: x.replace('excluded ', ''))
# where stain is elution or native set stain to NaN
df_stains.loc[df_stains['stain'].isin(['elution', 'native']), 'stain'] = np.nan
# delete stain rows where stain is hoechst
df_stains = df_stains[~df_stains['stain'].isin(['hoechst'])]
# write stain metadata to csv
df_stains.to_csv(Path(dir_4i_retina_example, 'stain_metadata.csv'), index=False)

# for each well_id create a directory and move files to this directory
for well_id in df['well_id'].unique():
    dir_well = Path(dir_4i_retina_example, 'stitched',well_id)
    dir_well.mkdir(parents=True, exist_ok=True)
    for file in df[df['well_id'] == well_id]['file'].values:
        os.system(f'mv stitched/{file} {dir_well}')

# load default config file
config = load_default_config()
# add regex to config
config['regex'] = regex_retina
# add path to stain metadata to config
config['stain_metadata'] = str(Path(dir_4i_retina_example, 'stain_metadata.csv'))
# set nuclei channel
config['channel_nuclei'] = 2
# set staining channels
config['channel_stains'] = [0, 1, 3]
# set stain nuclei
config['stain_nuclei'] = 'hoechst'

# save example config file
with open(Path(dir_4i_retina_example, 'config.yaml'), 'w') as file:
    yaml.dump(config, file, sort_keys=False)

# run phenoscapes
for well_id in df['well_id'].unique():
    process_sample(sample=well_id,
                   dir_input=dir_4i_retina_example,
                   dir_output=dir_4i_retina_example,
                   metadata_file=Path(dir_4i_retina_example,'stain_metadata.csv'),
                   config_file=Path(dir_4i_retina_example,'config.yaml'))
