from pathlib import Path
import pandas as pd
from skimage import io
from phenoscapes.utils import scale_image, get_metadata, annotate_img
from skimage.util import montage
import numpy as np
from skimage.transform import rescale

def generate_overview_montage(sample,
                              dir_images:str,
                              dir_masks:str,
                              dir_output:str,
                              df_stains:pd.DataFrame,
                              stains=None,
                              slice_step=5000,
                              config:dict = None):

    dir_images_sample = Path(dir_images,sample)
    df_images = get_metadata(dir_images_sample, custom_regex=config['regex'])
    df_images = pd.merge(df_images,df_stains,how='left')
    df_images.loc[df_images["channel_id"] == 0, "stain"] = 'DAPI'
    df_images['stain'] = df_images['stain'].fillna('bg_cycle')
    df_images = df_images.sort_values(['cycle_id', 'channel_id'], ascending=[True, True])
    if stains is not None:
        df_images = df_images['stain'].isin(stains)
    file_mask = Path(dir_masks, sample+'.tif')
    mask = io.imread(file_mask)
    center = np.round([np.average(indices) for indices in np.where(mask > 0)]).astype(int)
    imgs_full = []
    imgs_zoom_in = []
    for index,row in df_images.iterrows():
        annotation = ' '.join(['cycle ' + str(row['cycle_id']), 'channel ' + str(row['channel_id']), row['stain']])
        file_img = Path(dir_images_sample, row['file'])
        img = io.imread(file_img)
        img = scale_image(img)
        img_zoom_in = img[center[0]-2500:center[0]+2500,center[1]-2500:center[1]+2500]
        step = int(round(img.shape[0]/slice_step))
        img = img[::step,::step]
        img = annotate_img(img, annotation)
        img = rescale(img, 0.4, anti_aliasing=True)
        # remove last axis
        img = np.squeeze(img,-1)
        imgs_full.append(img)
        img_zoom_in = annotate_img(img_zoom_in, annotation)
        img_zoom_in = rescale(img_zoom_in, 0.4, anti_aliasing=True)
        img_zoom_in = np.squeeze(img_zoom_in,-1)
        imgs_zoom_in.append(img_zoom_in)

    img_montage_zoom = montage(imgs_zoom_in)
    img_montage_full = montage(imgs_full)

    # convert to uint8
    img_montage_full = (scale_image(img_montage_full,range=(0,1), percentile=0) * 255).astype(np.uint8)
    img_montage_zoom = (scale_image(img_montage_zoom,range=(0,1),percentile=0) * 255).astype(np.uint8)

    dir_output_sample = Path(dir_output,sample)
    dir_output_sample.mkdir(parents=True, exist_ok=True)
    io.imsave(Path(dir_output_sample,'montage_zoom_in.png'), img_montage_zoom, check_contrast=True)
    io.imsave(Path(dir_output_sample,'montage_full.png'), img_montage_full, check_contrast=True)
