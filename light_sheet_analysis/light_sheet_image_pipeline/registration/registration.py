import itk
import numpy as np
from skimage import exposure


def rigid_registration_image(fixed_image, moving_image):
    parameter_object = itk.ParameterObject.New()
    # Positions
    default_translation_parameter_map = parameter_object.GetDefaultParameterMap(
        "translation"
    )
    # postions

    # default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("translation",3)
    parameter_object.AddParameterMap(default_translation_parameter_map)
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        number_of_threads=64,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    return result_image, result_transform_parameters


def rescale_intensity(img):
    vmin, vmax = np.percentile(img, q=(1, 99))
    img = exposure.rescale_intensity(img, in_range=(vmin, vmax), out_range=np.float32)
    return img
