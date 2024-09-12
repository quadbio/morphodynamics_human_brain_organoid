from skimage.filters import gaussian, threshold_otsu
from skimage import img_as_uint
import matplotlib.pyplot as plt
from skimage import io
from skimage import measure
from scipy import ndimage
from pathlib import Path
from skimage.filters.rank import median
from skimage.morphology import disk
import numpy as np
from phenoscapes.utils import scale_image, get_metadata
import pandas as pd
# TODO: unify both masking functions as they are mainly the same!
def simple_mask(sample:str,
                dir_input:str,
                dir_output:str,
                cycle:int = 1,
                channels:list = [0,1,2,3],
                sigma:int = 100,
                n_binary:int = 50,
                outlier_threshold:int = 30,
                smooth_area:bool = True,
                save:bool = True,
                plot:bool = True,
                show_plot:bool = False,
                save_plot:bool = True,
                config:dict = None):

  """ Creating an initial simple mask

      :param config:
      :param sigma:
      :param n_binary:
      :param outlier_threshold:
      :param smooth_area:
      :param sample: (str) id of sample/region
      :param dir_input: (str) path to input directory
      :param dir_output: (str) path to output directory
      :param cycle: (str) cycle to create mask from
      :param channels: (list) channels to create mask from
      :param save: (bool) save created mask
      :param plot: (bool) create plot
      :param save_plot: (bool) save plot
      :param show_plot: (bool) return plot

      :return: depending on input parameters the function saves the created mask and/or plots and/or shows the overview plot
  """

  dir_images = Path(dir_input, sample)
  df = get_metadata(dir_images, custom_regex=config['regex'])

  # List images
  # filter cycle
  df = df[df['cycle_id'] == cycle]
  # filter channels
  df = df[df['channel_id'].isin(channels)]
  # list files
  files_images = df[df['cycle_id'] == cycle]['file'].values

  # Load image
  print('Loading images...')
  img_init = np.dstack([io.imread(Path(dir_images,file_image)) for file_image in files_images])
  print('Scaling images...')
  for i in range(img_init.shape[2]):
    img_init[...,i] = scale_image(img_init[...,i])
  print('Initial masking...')
  # max all channels
  img_init = np.max(img_init, axis=2)

  # Apply gaussian
  img = gaussian(img_init, sigma=sigma)

  # Otsu thresholding
  thr = threshold_otsu(img)
  img = (img > thr).astype(int)

  # Fill holes
  img = ndimage.binary_fill_holes(img).astype('uint8')

  # Median filter to remove outliers
  img = median(img, disk(outlier_threshold))

  # Select biggest area
  img = measure.label(img)
  regions = measure.regionprops(img)
  regions.sort(key=lambda x: x.area, reverse=True)
  if len(regions) > 1:
      for rg in regions[1:]:
         img[rg.coords[:,0], rg.coords[:,1]] = 0

  if smooth_area:
    print('Final masking...')
    # Smoothing selected area
    # Dilation
    struct = ndimage.generate_binary_structure(2, 2)
    img = ndimage.morphology.binary_dilation(img, structure=struct, iterations=n_binary).astype(int)
    # Apply gaussian
    img = gaussian(img, sigma=int(sigma/2))
    # Apply erosion
    img = ndimage.morphology.binary_erosion(img, structure=struct, iterations=n_binary).astype(int)
    # Otsu thresholding
    thr = threshold_otsu(img)
    img = (img > thr).astype(int)

    # Fill holes
    img = ndimage.binary_fill_holes(img).astype(int)
  # Save mask
  if save:
    print('Saving mask...')
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    io.imsave(Path(dir_output,f'{sample}.tif'), img_as_uint(img), check_contrast=False)

  # Plot mask
  if plot:
    print('Plotting mask overviews...')
    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

    ax[0].imshow(img_init)
    ax[0].axis('off')
    ax[0].set_title('Max over all channels')

    ax[1].imshow(img)
    ax[1].axis('off')
    ax[1].set_title('Mask')

    ax[2].imshow(img_init*img)
    ax[2].axis('off')
    ax[2].set_title('Masked image')
    if save_plot:
      Path(dir_output,'plots').mkdir(parents=True, exist_ok=True)
      plt.savefig(Path(dir_output,'plots', f'{sample}.png'))
    if show_plot:
      plt.show()


def refined_mask(sample:str,
                dir_input:str,
                dir_output:str,
                df_stains:pd.DataFrame,
                config:dict,
                sigma=100,
                n_binary:int = 50,
                save:bool = True,
                plot:bool = True,
                show_plot:bool = False,
                save_plot:bool = True,
                use_only_dapi: bool = False):
  """ Refining initial simple mask

      :param config:
      :param df_stains:
      :param sample: (str) id of sample/region
      :param dir_input: (str) path to input directory
      :param dir_output: (str) path to output directory
      :param sigma: (int) sigma
      :param n_binary:
      :param save: (bool) save created mask
      :param plot: (bool) create plot
      :param save_plot: (bool) save plot
      :param show_plot: (bool) return plot

      :return: depending on input parameters the function saves the created mask and/or plots and/or shows the overview plot
  """
  print('Refining mask...')
  # extract parameters
  # filter df_stains for NaN
  df_stains = df_stains[df_stains['stain'].notna()]
  cycles = df_stains['cycle_id'].unique().tolist()
  cycle_dapi = config['reference_cycle']
  # List images
  dir_images = Path(dir_input, sample)
  df = get_metadata(dir_images, custom_regex=config['regex'])
  file_dapi = df[(df['cycle_id'] == cycle_dapi) & (df['channel_id'] == 0)]['file'].values
  if use_only_dapi:
    files_images=file_dapi
  else:
    df = df[df['cycle_id'].isin(cycles)]
    df = df[df['channel_id'].isin(config['channel_stains'])]
    files_images = df['file'].values
    files_images = np.concatenate([file_dapi,files_images])

  # Load image
  img_init = np.dstack([io.imread(Path(dir_images,file_image)) for file_image in files_images])

  for i in range(img_init.shape[2]):
    img_init[...,i] = scale_image(img_init[...,i])

  # max all channels
  img_init = np.max(img_init, axis=2)

  # Apply gaussian
  img = gaussian(img_init, sigma=sigma)

  # Otsu thresholding
  thr = threshold_otsu(img)
  img = (img > thr).astype(int)

  # Fill holes
  img = ndimage.binary_fill_holes(img).astype('uint8')

  # Median filter to remove outliers
  img = median(img, disk(30))

  # Select biggest area
  img = measure.label(img)
  regions = measure.regionprops(img)
  regions.sort(key=lambda x: x.area, reverse=True)
  if len(regions) > 1:
      for rg in regions[1:]:
         img[rg.coords[:,0], rg.coords[:,1]] = 0

  # Smoothing selected area
  # Dilation
  struct = ndimage.generate_binary_structure(2, 2)
  img = ndimage.morphology.binary_dilation(img, structure=struct, iterations=n_binary).astype(int)

  # Apply gaussian
  img = gaussian(img, sigma=int(sigma/2))

  # Otsu thresholding
  thr = threshold_otsu(img)
  img = (img > thr).astype(int)

  # Fill holes
  img = ndimage.binary_fill_holes(img).astype(int)

  # Save mask
  if save:
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    io.imsave(Path(dir_output,sample+'.tif'), img_as_uint(img), check_contrast=False)

  # Plot mask
  if plot:
    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

    ax[0].imshow(img_init)
    ax[0].axis('off')
    ax[0].set_title('Max over all channels')

    ax[1].imshow(img)
    ax[1].axis('off')
    ax[1].set_title('Mask')

    ax[2].imshow(img_init*img)
    ax[2].axis('off')
    ax[2].set_title('Masked image')
    if save_plot:
      Path(dir_output,'plots').mkdir(parents=True, exist_ok=True)
      plt.savefig(Path(dir_output,'plots', sample+'.png'))
    if show_plot:
      plt.show()