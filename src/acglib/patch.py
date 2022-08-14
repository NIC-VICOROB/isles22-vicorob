import copy
import itertools
import timeit

import numpy as np

import sys

from .lists import resample_regular


def get_patch_span(shape):
    """Returns the patch span (before_C, after_C) for each axis with respect to the patch center C"""
    return tuple((int(np.floor((s - 1.) / 2.)), int(np.ceil((s - 1.) / 2.)) + 1) for s in shape)


def get_patch_slices(total_ndims, center, shape):
    # Compute the patch span around the center
    span = get_patch_span(shape)
    # Get slices for trailing_dims, of which we take all dims (i.e. all the batch, all the channels...)
    trailing_dim_slices = (total_ndims - len(center)) * (slice(None),)
    # Get slices for patched dimensions
    patched_dim_slices = tuple(slice(int(c_i) - sp_i[0], int(c_i) + sp_i[1]) for c_i, sp_i in zip(center, span))
    return trailing_dim_slices + patched_dim_slices


def get_patch(image, center, shape):
    """Returns an image patch with the requested shape around the given center.
    If the image has more dimensions than the given shape, only the last n dimensions will be sliced.
    If a dimension from shape is even, the center will be offset by -1 in that dimension.
    """
    return copy.deepcopy(image[get_patch_slices(image.ndim, center, shape)])


def clip_centers_inside_bounds(centers, image_shape, patch_shape):
    """Clips centers so the extracted patch does not fall out of image bounds"""
    clipped_centers = np.clip(a=centers,
                              a_min=np.ceil(np.divide(patch_shape, 2.0)).astype(int),
                              a_max=image_shape - (np.ceil(np.divide(patch_shape, 2.0)).astype(int) + 1))
    return clipped_centers.tolist()


def add_random_offset_to_centers(centers, image_shape, patch_shape):
    """Adds a random offset of up to half the patch_shape and clips them so patch is not outside image bounds."""
    if len(image_shape) > len(patch_shape):
        image_shape = image_shape[:len(patch_shape)]
    # Add random offset to centers
    np.random.seed(0)  # Repeatability
    offset_range = np.divide(patch_shape, 2).astype(int)
    offset_values = offset_range * ( ( np.random.random_sample((len(centers), len(centers[0]))) - 0.5 ) * 2.0 )
    centers = np.add(centers, np.rint(offset_values).astype(int))
    # Clip so none is out of bounds
    return clip_centers_inside_bounds(centers, image_shape, patch_shape)


def sample_centers_uniform(
    image_shape,
    step,
    patch_shape,
    mask=None,
    num_centers=None,
    add_rand_offset=False,
    ensure_full_coverage=False
):
    assert len(image_shape) == len(patch_shape) == len(step), f'{image_shape}, {patch_shape}, {step}'
    # Get patch span from the center in each dimension
    span = get_patch_span(patch_shape)
    # Compute stop_offset for the case of full_image_coverage
    stop_offset = step if ensure_full_coverage else (0,) * len(step)
    # Generate the sampling indexes for each dimension first and then get all their combinations (itertools.product)
    dim_indexes = [list(range(sp[0], ims - sp[1] + 1 + so, st)) 
                   for sp, ims, st, so in zip(span, image_shape, step, stop_offset)]
    centers = list(itertools.product(*dim_indexes))
    # If mask is given, keep centers where mask is nonzero
    if mask is not None:
        assert np.array_equal(image_shape, mask.shape)
        centers = [c for c in centers if mask[tuple(c)] != 0.0]
    # Resample to target number of centers
    if num_centers is not None:
        centers = resample_regular(centers, num_centers)
    # Add random offset if required
    if add_rand_offset:
        centers = add_random_offset_to_centers(centers, image_shape, patch_shape)
    return clip_centers_inside_bounds(centers, image_shape, patch_shape)


def sample_centers_mask(mask_image, patch_shape, n, add_rand_offset=False):
    assert mask_image.ndim < 4
    # Sample all flat indexes where mask is greater than 0
    center_indexes = list(np.flatnonzero(mask_image > 0.0))
    # Resample to target number of centers
    center_indexes = resample_regular(center_indexes, n)
    # Go from flat index to 2d/3d coordinates
    centers = np.transpose(np.unravel_index(center_indexes, shape=mask_image.shape)).tolist()
    # Add random offset if required
    if add_rand_offset:
        centers = add_random_offset_to_centers(centers, mask_image.shape, patch_shape)
    return clip_centers_inside_bounds(centers, mask_image.shape, patch_shape)


def sample_centers_balanced(labels_image, patch_shape, n, add_rand_offset=False, exclude=None):
    # Get labels from which to extract centers
    label_ids = np.unique(labels_image)
    if exclude is not None:
        label_ids = [i for i in label_ids if i not in exclude]
    # Compute the amount of centers per label
    centers_per_label = int(np.floor(n / len(label_ids)))
    # Sample all centers from each label, then regular resample to target goal
    centers = []
    for label_id in label_ids:
        label_indexes = list(np.flatnonzero(labels_image == label_id))
        label_indexes = resample_regular(label_indexes, centers_per_label)
        label_centers = np.transpose(np.unravel_index(label_indexes, shape=labels_image.shape)).tolist()
        centers += label_centers
    # Add random offset if required
    if add_rand_offset:
        centers = add_random_offset_to_centers(centers, labels_image.shape, patch_shape)
    # Clip to bounds and return
    return clip_centers_inside_bounds(centers, labels_image.shape, patch_shape)


def sample_centers_labels_fraction(labels_image, labels_fractions, patch_shape, n, add_rand_offset=False):
    # Compute the amount of centers per label
    labels_num_centers = {lid: int(np.floor(n * lfraction)) for lid, lfraction in labels_fractions.items()}

    # Sample all centers from each label, then regular resample to target goal
    centers = []
    for label_id, label_num_centers in labels_num_centers.items():
        label_indexes = list(np.flatnonzero(labels_image == label_id))
        label_indexes = resample_regular(label_indexes, label_num_centers)
        label_centers = np.transpose(np.unravel_index(label_indexes, shape=labels_image.shape)).tolist()
        centers += label_centers
    # Add random offset if required
    if add_rand_offset:
        centers = add_random_offset_to_centers(centers, labels_image.shape, patch_shape)
    # Clip to bounds and return
    return clip_centers_inside_bounds(centers, labels_image.shape, patch_shape)


def sample_centers_labels_number(labels_image, labels_num_centers, patch_shape, add_rand_offset=False):
    # Sample all centers from each label, then regular resample to target goal
    centers = []
    for label_id, label_num_centers in labels_num_centers.items():
        label_indexes = list(np.flatnonzero(labels_image == label_id))
        if len(label_indexes) > 0 and label_num_centers > 0:
            label_indexes = resample_regular(label_indexes, label_num_centers)
            label_centers = np.transpose(np.unravel_index(label_indexes, shape=labels_image.shape)).tolist()
        else:
            label_centers = []
        centers += label_centers
    # Add random offset if required
    if add_rand_offset:
        centers = add_random_offset_to_centers(centers, labels_image.shape, patch_shape)
    # Clip to bounds and return
    return clip_centers_inside_bounds(centers, labels_image.shape, patch_shape)


def sample_centers_weighted(weights_image, num_centers, patch_shape, add_rand_offset=False):
    """Samples center coordinates for patch extraction with likelihood proportional to the weights"""
    # Compute the cumulative weight space for sampling
    w_accum = np.cumsum(weights_image)
    # Generate n sampling points, each ranging from 0 to the maximum weight
    sampling_points = np.random.random_sample((num_centers,)) * w_accum[-1]
    # Sample the weight space using the searchsorted numpy function to get binary search in space
    sampled_flat_indexes = np.searchsorted(w_accum, sampling_points, side='left')
    # Obtain centers from flat indexes
    centers = np.transpose(np.unravel_index(sampled_flat_indexes, shape=weights_image.shape))
    # Add random offset if required
    if add_rand_offset:
        centers = add_random_offset_to_centers(centers, weights_image.shape, patch_shape)
    # Clip so not out of bounds
    return clip_centers_inside_bounds(centers, weights_image.shape, patch_shape)


if __name__ == '__main__':
    a = np.zeros((5,5))
    a[2, 2] = 1

    # print(sample_centers_uniform(
    #     image_shape=(5, 5), step=(2,2), patch_shape=(4,4), mask=None, max_centers=None))