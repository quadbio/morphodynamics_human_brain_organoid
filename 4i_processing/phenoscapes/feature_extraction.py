import numpy as np
from skimage import io
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from pathlib import Path
from phenoscapes.utils import get_metadata
import pandas as pd
from tqdm import tqdm
#TODO: add possibilty for other measures like morpholgy, hue moments etc.
def intensity_median(regionmask, intensity):
    return np.median(intensity[regionmask])


def extract_features(sample:str,
                     dir_images:str,
                     dir_segmented:str,
                     dir_output:str,
                     df_stains:pd.DataFrame,
                     config:dict = None):

    dir_images_sample = Path(dir_images, sample)
    df_images = get_metadata(dir_images_sample, custom_regex=config['regex'])
    df_images = pd.merge(df_images, df_stains, how='left')
    df_images.loc[df_images["channel_id"] == config['channel_nuclei'], "stain"] = config['stain_nuclei']
    df_images['stain'] = df_images['stain'].fillna('bg_cycle')
    stains = df_images['stain'].values
    # create stain dictionary from stains keys are numbers, values are stain names
    stain_dict = {i: stain for i, stain in enumerate(stains)}

    imgs = np.dstack([io.imread(Path(dir_images_sample, file)) for file in tqdm(df_images['file'].values)])

    img_label = io.imread(Path(dir_segmented, sample + '.tif'))
    
    if config['feature_extraction']['expand_labels'] != None:
        img_label=expand_labels(img_label,config['feature_extraction']['expand_labels'])

    df = pd.DataFrame(regionprops_table(label_image=img_label,
                                        intensity_image=imgs,
                                        properties=('label','centroid','area', 'intensity_mean','intensity_max','intensity_min'),
                                        extra_properties=(intensity_median,)))

    # update column names
    for col in df.columns:
        if col.startswith('intensity'):
            stain_id = int(col.split('-')[-1])
            praefix = col.split('-')[0]
            df.rename(columns={col: f'{praefix}_{stain_dict[stain_id]}'}, inplace=True)

    # add sample column
    df['sample'] = sample

    # save to csv
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(dir_output, sample + '.csv'), index=False)

