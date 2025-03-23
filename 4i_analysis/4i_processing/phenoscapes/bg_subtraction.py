import pandas as pd
from skimage import io
from pathlib import Path
from phenoscapes.utils import get_metadata
import numpy as np
from scipy.optimize import nnls
from tqdm import tqdm

def run_bg_subtraction(sample: str,
                       dir_input: str,
                       dir_masks: str,
                       dir_speckle_masks:str,
                       dir_output: str,
                       df_stains: pd.DataFrame = None,
                       config:dict = None):
    """
    Run background subtraction on a given sample.
    :param config:
    :param df_stains:
    :param sample:
    :param dir_input:
    :param dir_masks:
    :param dir_output:
    :return:
    """

    cycles_bg = config['bg_subtraction']['bg_cycles']
    cycles = config['bg_subtraction']['cycles']
    cycle_dapi = config['reference_cycle']

    if cycles is None and cycles_bg is None:
        if df_stains is not None:
            cycles_bg = df_stains[df_stains.isna().any(axis=1)]['cycle_id'].unique()
            cycles = df_stains[~df_stains.isna().any(axis=1)]['cycle_id'].unique()

    # Get metadata, set paths and create directories
    dir_images = Path(dir_input, sample)
    df = get_metadata(dir_images, custom_regex=config['regex'])
    dir_output = Path(dir_output, sample)
    dir_output.mkdir(parents=True, exist_ok=True)

    # Get file names of reference dapi and all other channels/cycles
    file_dapi = df[(df['cycle_id'] == cycle_dapi) & (df['channel_id'] == config['channel_nuclei'])]['file'].values[0]
    df_bg = df[df['cycle_id'].isin(cycles_bg)]
    df = df[df['cycle_id'].isin(cycles)]
    df = df[df['channel_id'].isin(config['channel_stains'])]
    files_images = df['file'].values

    # Load and save dapi image
    img = io.imread(Path(dir_images, file_dapi))
    io.imsave(Path(dir_output, file_dapi), img, check_contrast=False)

    # Load mask
    mask = io.imread(Path(dir_masks, sample + '.tif'))

    for file_image in tqdm(files_images):
        img = io.imread(Path(dir_images, file_image))
        cycle = df[df['file'] == file_image]['cycle_id'].values[0]
        channel = df[df['file'] == file_image]['channel_id'].values[0]
        cycles_bg_select = np.abs(cycle - np.array(cycles_bg)).argsort()
        cycle_closer = int(np.array(cycles_bg)[cycles_bg_select[0]])
        cycle_further = int(np.array(cycles_bg)[cycles_bg_select[1]])
        cycles_sorted = sorted([cycle_closer, cycle_further])
        leading_cycle_bg = cycles_sorted[0]
        file_leading_cycle_bg = df_bg[(df_bg['cycle_id'] == leading_cycle_bg) &
                                      (df_bg['channel_id'] == channel)]['file'].values[0]
        img_bg_leading = io.imread(Path(dir_images, file_leading_cycle_bg))
        if cycle < cycles_bg.min():
            # skip bg subtraction
            print(f'Skipping bg subtraction for cycle {cycle} as no previous bg cycle exists. Earliest found bg cycle is {cycles_bg.min()}.')
            io.imsave(Path(dir_output, file_image), img, check_contrast=False)

        else:
            lagging_cycle_bg = cycles_sorted[1]
            file_lagging_cycle_bg = df_bg[(df_bg['cycle_id'] == lagging_cycle_bg) &
                                          (df_bg['channel_id'] == channel)]['file'].values[0]
            img_bg_lagging = io.imread(Path(dir_images, file_lagging_cycle_bg))
            if cycle > cycles_bg.max():
                print(f'No lagging bg cycle found for cycle {cycle}. Using only {lagging_cycle_bg} instead.')
                img_bg = img_bg_lagging.copy()
            else:
                cycle_range = lagging_cycle_bg - leading_cycle_bg
                factor_leading = (cycle_range - (cycle - leading_cycle_bg)) / cycle_range
                factor_lagging = (cycle_range - (lagging_cycle_bg - cycle)) / cycle_range
                img_bg = (img_bg_lagging * factor_lagging) + (img_bg_leading * factor_leading)

            bg_factor = nnls(img_bg[mask == 1][..., np.newaxis], img[mask == 1])[0]
            img_bg = img_bg * bg_factor
            img = img - img_bg
            img[img < 0] = 0
            img = img.astype(np.uint16)
            if config['speckle_removal']['run']:
                dir_speckle_mask = Path(dir_speckle_masks,sample)
                speckle_mask=io.imread(Path(dir_speckle_mask, file_image))
                io.imsave(Path(dir_output, file_image), speckle_mask*img, check_contrast=False)
            else:
                io.imsave(Path(dir_output, file_image), img, check_contrast=False)
