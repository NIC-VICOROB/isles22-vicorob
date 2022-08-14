import numpy as np

def get_crop_border_slice(image, background_value=0):
    if not isinstance(image, np.ndarray):
        image = np.asanyarray(image)
    foreground = (image != background_value)

    crop_slice = []
    for dim_idx in range(image.ndim):
        # Compact array to a single axis to make np.argwhere much more efficient
        compact_axis = tuple(ax for ax in range(image.ndim) if ax != dim_idx)
        foreground_indxs = np.argwhere(np.max(foreground, axis=compact_axis) == True)
        # Find the dimensions lower and upper foreground indices
        crop_slice.append(slice(np.min(foreground_indxs), np.max(foreground_indxs) + 1))
    return tuple(crop_slice)


def crop_borders(image, background_value=0):
    """Crops the background borders of an image. If any given channel is all background, it will also be cropped!

    :param image: the image to crop.
    :param background_value: value of the background that will be cropped.
    :return: The image with background borders cropped.
    """
    crop_slice = get_crop_border_slice(image, background_value)
    return image[crop_slice]

