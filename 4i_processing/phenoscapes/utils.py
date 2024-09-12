# function collection for pancreas primary image analysis pipeline
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
# TODO: check if custom regex is working
def get_metadata(dir_images, custom_regex=None):

  images = os.listdir(dir_images)
  images = [image for image in images if '.tif' in image]
  if custom_regex is not None:
    regex = custom_regex
  else:
    regex = r'_c(?P<cycle_id>\d{3})_(?P<well_id>[A-Z]\d{2})_r(?P<region_id>\d{3})_t(?P<timeline_id>\d{3})_z\d{3}_c\d{1}_stitch_ch(?P<channel_id>\d{2})\.tif$'
  df = pd.DataFrame({'file': images})
  df = df.join(df['file'].str.extractall(regex, flags=re.IGNORECASE).groupby(level=0).last())
  # remove rows where cycle_id  region_id or channel_id is NaN
  df = df.dropna()
  df['cycle_id'] = df['cycle_id'].apply(lambda x: int(x))
  df['channel_id'] = df['channel_id'].apply(lambda x: int(x))
  return df

def crop_image(img, mask):
  mask = mask > 0
  return img[np.ix_(mask.any(1), mask.any(0))]

def denoise_image(img):

  sigma_est = np.mean(estimate_sigma(img, channel_axis=None))

  patch_kw = dict(patch_size=5,  # 5x5 patches
                  patch_distance=6,  # 13x13 search area
                  channel_axis=None) # TODO: deprecated syntax -> to new syntax

  img = denoise_nl_means(img, h=0.8 * sigma_est,
                         sigma=sigma_est, preserve_range=True,
                         fast_mode=True, **patch_kw)
  return img

def scale_image(image, percentile=1, range=(0,65535)):
  image = np.interp(image, (np.percentile(image,percentile), np.percentile(image,100 - percentile)), range)
  return image

def show_image(image, scale=True, grayscale=False, adapt_figsize=True, factor_adapt=200, sample=None):
  if sample is not None:
    image = image[::sample, ::sample]
  if scale:
    image = scale_image(image)
  if adapt_figsize:
    width = round(image.shape[1]/factor_adapt)
    height = round(image.shape[0]/factor_adapt)
    plt.figure(figsize=(width,height))
  else:
    plt.figure(figsize=(20, 20))
  if grayscale:
    plt.imshow(image, cmap=plt.cm.gray)
  else:
    plt.imshow(image)
  plt.show()

def overlay_channels(img, img_2, img_3=None):
  img = img/65535
  img_2 = img_2/65535
  if img_3 is None:
    img_3 = np.zeros(img.shape)
  else:
    img_3 = img_3/65535
  return np.dstack([img, img_2, img_3])

def annotate_img(img, annotation, size=128, x=0,y=0):
  fig = plt.figure()
  fig.figimage(
    img,
    resize=True,  # Resize the figure to the image to avoid any interpolation.
  )
  fig.text(x, y, annotation, fontsize=size, color='white', va="bottom")
  canvas = plt.gca().figure.canvas
  canvas.draw()
  data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
  annotated_img = data.reshape(canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return (annotated_img)

# TODO: add scale bar option for plots which are made! -> simple mask, refined mask, overlays, montages
def add_scale_bar(img, scale=1, bar_length=100, offset_x=100, off_set_y=100,bar_thickness=10, color=(1, 1, 1)):
    # if image is not rgb, convert to rgb
    if len(img.shape) == 2:
      img = np.stack((img, img, img), axis=2)
    bar_length = bar_length * scale
    bar_thickness = bar_thickness * scale
    bar_end_x = img.shape[1] - offset_x
    bar_start_x = bar_end_x - bar_length
    bar_start_y = img.shape[0] - off_set_y - bar_thickness
    bar_end_y = bar_start_y - bar_thickness
    img[bar_end_y:bar_start_y, bar_start_x:bar_end_x, :] = color
    return img

def convert_uint16(img):
    if img.max() > 65535:
        print('Warning: image contains values higher than 65535. Values will be clipped to 65535.')
        img[img > 65535] = 65535
    if img.min() < 0:
        print('Warning: image contains values lower than 0. Values will be clipped to 0.')
        img[img < 0] = 0
    img = img.astype('uint16')
    return img