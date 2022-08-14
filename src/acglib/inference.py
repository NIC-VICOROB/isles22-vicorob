import copy

import torch
import numpy as np

import os

import sys

from .generators import InstructionDataset, construct_dataloader
from .patch import sample_centers_uniform, get_patch_slices

from .time_utils import RemainingTimeEstimator
from .print_utils import print_progress_bar


def inference_image_patches(
        image,
        model,
        patch_shape_in,
        patch_shape_out,
        step,
        batch_size,
        device,
        extract_patch_func,
        postprocess_patch_func=None,
        verbose=True):
    x_orig = copy.deepcopy(image)

    image : np.ndarray
    assert image.ndim == len(patch_shape_in) == len(patch_shape_out) == 4
    assert len(step) == 3

    # First pad image to ensure all voxels in volume are processed independently of extraction_step
    pad_dims = [(0,)] + [(int(np.ceil(in_dim / 2.0)),) for in_dim in patch_shape_in[1:]]
    x = np.pad(image, pad_dims, mode='edge')

    # Create patch generator with known patch center locations.
    patch_centers = sample_centers_uniform(x.shape[1:], step, patch_shape_in[1:])
    patch_slices = [get_patch_slices(len(patch_shape_in), center, patch_shape_in[1:]) for center in patch_centers]

    patch_gen = construct_dataloader(
        dataset=InstructionDataset(
            instructions=patch_centers, data=x, get_item_func=extract_patch_func),
        batch_size=batch_size,
        shuffle=False)

    # Put accumulation in torch (GPU accelerated :D)
    num_ch_out = patch_shape_out[0]
    voting_img = torch.zeros((num_ch_out,) + x[0].shape, device=device).float()
    counting_img = torch.zeros_like(voting_img).float()

    if postprocess_patch_func is None:
        postprocess_model_output = lambda _ : _
    else:
        postprocess_model_output = postprocess_patch_func

    old_stdout = sys.stdout  # backup current stdout
    if not verbose: # Make print functions not write to terminal
        sys.stdout = open(os.devnull, "w")

    # Perform inference and accumulate results in torch (GPU accelerated :D (if device is cuda))
    model.eval()
    model.to(device)
    with torch.no_grad():
        rta = RemainingTimeEstimator(len(patch_gen))

        for n, (x_patch, x_slice) in enumerate(zip(patch_gen, patch_slices)):
            x_patch = x_patch.to(device)
            y_pred = model(x_patch)
            y_pred = postprocess_model_output(y_pred)

            batch_slices = patch_slices[batch_size * n:batch_size * (n + 1)]
            for predicted_patch, patch_slice in zip(y_pred, batch_slices):
                voting_img[patch_slice] += predicted_patch
                counting_img[patch_slice] += torch.ones_like(predicted_patch)

            print_progress_bar( # If not verbose, stdout is redirected to devnull so nothing is printed
                    iteration=batch_size * n,
                    total=batch_size * len(patch_gen),
                    suffix=f'patches - ETA: {rta.update(n)}',
                    length=20)

        print_progress_bar( # If not verbose, stdout is redirected to devnull so nothing is printed
                iteration=batch_size * len(patch_gen),
                total=batch_size * len(patch_gen),
                suffix=f'patches - ETA: {rta.elapsed_time()}',
                length=20)

    # Restore original stdout if we changed it to avoid printing
    sys.stdout = old_stdout

    counting_img[counting_img == 0.0] = 1.0  # Avoid division by 0
    predicted_volume = torch.div(voting_img, counting_img).detach().cpu().numpy()
    
    # Unpad volume to return to original shape
    unpad_slice = \
        [slice(None)] + [slice(in_dim[0], x_dim - in_dim[0]) for in_dim, x_dim in zip(pad_dims[1:], x.shape[1:])]
    predicted_volume = predicted_volume[tuple(unpad_slice)]

    assert np.array_equal(x_orig.shape[1:], predicted_volume.shape[1:]), f'{x_orig.shape} != {predicted_volume.shape}'
    return predicted_volume
    
