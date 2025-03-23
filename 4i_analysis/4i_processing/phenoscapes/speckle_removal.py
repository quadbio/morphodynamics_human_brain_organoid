from skimage import io
from pathlib import Path
from phenoscapes.utils import get_metadata, crop_image
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter

def mask_speckles(df_channel:pd.DataFrame,
                  dir_images:str,
                  dir_output_masks:str,
                  cropping:bool,
                  mask:np.array,
                  percentile:int=2,
                  consecutive_count:int=4):
    """
    Run masking of high intensity speckles on a given sample and channel.
    :param df_channel:
    :param dir_images:
    :param dir_output_masks:
    :param crop:
    :param mask:
    :param percentile:
    :param consecutive_count:
    :return:
    """

    #Get top n-th percentile masks for each image
    percentile_masks=[]
    for index,row in df_channel.iterrows():
        file_img = Path(dir_images, row['file'])
        image=io.imread(file_img)
        image=gaussian_filter(image,5)
        img_perc=np.percentile(image,100 - percentile)
        image=(image>img_perc)
        percentile_masks.append(image)

    #Convert to int and sliding window --> find n-consecutive top percentile pixels
    percentile_masks=np.array(percentile_masks).astype(int)
    sliding_windows=np.lib.stride_tricks.sliding_window_view(percentile_masks,consecutive_count,axis=0)
    three_in_a_rows=sliding_windows.sum(-1)
    three_in_a_rows=three_in_a_rows==consecutive_count
    speckle_mask_array=np.ones(percentile_masks.shape)
    for i in range(percentile_masks.shape[1]):
        for j in range(percentile_masks.shape[2]):
            out_array=np.ones(percentile_masks.shape[0])
            three_in_a_row=three_in_a_rows[:,i,j]
            for ind in np.where(three_in_a_row)[0]:
                out_array[ind:ind+consecutive_count]=0
            speckle_mask_array[:,i,j]=out_array

    #save speckle_masks

    for row,speckle_mask in zip(df_channel.index,speckle_mask_array): 
        if cropping:
            io.imsave(Path(dir_output_masks, df_channel.loc[row]['file']),crop_image(speckle_mask , mask), check_contrast=False)
        else:
            io.imsave(Path(dir_output_masks, df_channel.loc[row]['file']),speckle_mask, check_contrast=False)

def mask_speckles_channels(df_cycle:pd.DataFrame,
                  dir_images:str,
                  dir_output_masks:str,
                  mask:np.array,
                  cropping:bool,
                  percentile:int=2):
    """
    Run masking of high intensity speckles on a given sample and channel.
    :param df_cycle:
    :param dir_images:
    :param dir_output_masks:
    :param crop:
    :param mask:
    :param percentile:
    :return:
    """

    #Get top n-th percentile masks for each image
    percentile_masks=[]
    for index,row in df_cycle.iterrows():
        file_img = Path(dir_images, row['file'])
        image=io.imread(file_img)
        image=gaussian_filter(image,5)
        img_perc=np.percentile(image,100 - percentile)
        image=(image>img_perc)
        percentile_masks.append(image)
    percentile_masks=np.array(percentile_masks).astype(int)
    #Find pixels that occure in the top n-th percentile for all stain channels
    speckle_mask=(percentile_masks.sum(0)<int(len(percentile_masks))).astype(int)
    #save speckle_masks    
    if cropping:
        speckle_mask=crop_image(speckle_mask,mask)
        
    for row in df_cycle.index: 
        speckle_mask_channel=io.imread(Path(dir_output_masks, df_cycle.loc[row]['file']))
        speckle_mask_row=speckle_mask_channel*speckle_mask
        io.imsave(Path(dir_output_masks, df_cycle.loc[row]['file']),speckle_mask_row, check_contrast=False)


def run_speckle_removal(sample,
                               dir_input: str,
                               dir_masks: str,
                               dir_output_masks: str,
                               config: dict):
    """
    Run speckle removal on a given sample.
    :param sample:
    :param dir_input:
    :param dir_masks:
    :param dir_output_masks:
    :param cycle_dapi:
    :param n_processes:
    :param config:

    :return:
    """

    # Set paths and create directories
    #Create output dir
    out_dir = Path(dir_output_masks,sample)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    mask = io.imread(Path(dir_masks, sample + '.tif'))

    # Get metadata
    dir_images = Path(dir_input, sample)
    df_images = get_metadata(dir_images, custom_regex=config['regex'])
    df_images = df_images.sort_values(['cycle_id', 'channel_id'], ascending=[True, True])
    df_images=df_images[df_images['channel_id']!= 0]


    #Iterate over channel
    for channel in df_images['channel_id'].unique():
        df_channel=df_images[df_images['channel_id']==channel]
        mask_speckles(df_channel=df_channel,
                      dir_images=dir_images,
                      dir_output_masks=out_dir,
                      mask=mask,
                      percentile=config['speckle_removal']['percentile'],
                      cropping=config['cropping_denoising']['cropping'],
                      consecutive_count=4)
    #Iterate over cycles
    for cycle in df_images['cycle_id'].unique():
        df_cycle=df_images[df_images['cycle_id']==cycle]
        mask_speckles_channels(df_cycle=df_cycle,
                      dir_images=dir_images,
                      dir_output_masks=out_dir,
                      mask=mask,
                      cropping=config['cropping_denoising']['cropping'],
                      percentile=config['speckle_removal']['percentile'])
