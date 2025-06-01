'''
Attenuation filter function extracted from DEXP package.

Originally from the projection function of the great DEXP package:
https://github.com/royerlab/dexp/blob/8e8399f5d0d8f1e1ae0ddfa6cb6011921929ae0b/dexp/processing/color/projection.py#L127

Copyright (c) 2021, DEXP
License: BSD 3-Clause (see LICENSE_dexp.txt)
Modifications: Extracted as standalone function, adapted for modular use
'''

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter, rotate


def attenuation_filter(
    image, attenuation_min_density, attenuation, attenuation_filtering
):
    """
    Apply attenuation filtering to image data.
    
    Extracted from DEXP's projection function for standalone use.
    
    Parameters:
    -----------
    image : cupy.ndarray
        Input image data
    attenuation_min_density : float
        Minimum density for attenuation calculation
    attenuation : float
        Attenuation coefficient
    attenuation_filtering : float
        Gaussian filter sigma for attenuation preprocessing
        
    Returns:
    --------
    cupy.ndarray
        Filtered image with attenuation applied
    """

    if attenuation_filtering > 0:
        image_for_attenuation = gaussian_filter(image, sigma=attenuation_filtering)
    else:
        image_for_attenuation = image

    cum_density = cp.cumsum(
        attenuation_min_density + (1 - attenuation_min_density) * image_for_attenuation,
        axis=0,
    )

    image *= cp.exp(-attenuation * cum_density)
    return image