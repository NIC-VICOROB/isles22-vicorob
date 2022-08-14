import os
import numpy as np
import copy
import nibabel as nib
import SimpleITK as sitk
from utils_simpleitk import register_affine_sitk, transform_sitk_image

def compute_normalization_parameters(arr, norm_type, ignore_background_label=None):
    # Copy image to avoid editing original arr
    norm_image = copy.deepcopy(arr)
    if ignore_background_label is not None:
        norm_image[norm_image == ignore_background_label] = np.nan

    if norm_type == 'z_score':
        norm_params = (
            np.nanmean(norm_image, axis=(-3, -2, -1), keepdims=True),
            np.nanstd(norm_image, axis=(-3, -2, -1), keepdims=True))
    elif norm_type == 'percentile_minmax':
        norm_params = np.nanpercentile(norm_image, [0.05, 99.95], axis=(-3, -2, -1), keepdims=True)
    else:
        raise NotImplementedError(f'{norm_type} normalization not supported')
    
    return norm_params

def normalize_array(arr, norm_params, norm_type):
    if norm_type == 'z_score':
        arr = (arr - norm_params[0]) / norm_params[1]
    elif norm_type == 'percentile_minmax':
        new_low, new_high = norm_params
        arr = (arr - new_low) / (new_high - new_low)  # Put between 0 and 1
        arr = np.clip((2.0 * arr) - 1.0, -1.0, 1.0)  # Put between -1 and 1 and clip extrema
    else:
        raise NotImplementedError(f'{norm_type} normalization not supported')
    
    return arr


def compute_symmetric_modalities(in_fpaths, out_fpaths):
    sym_tx_fpath = os.path.join(os.path.dirname(out_fpaths[0]), 'sym_tx.mat')

    # Then apply flip and transform to the others
    for n, (in_fpath, out_fpath) in enumerate(zip(in_fpaths, out_fpaths)):
        ### First flip and register the first one within itself
        nifti = nib.load(in_fpath)
        # Find the LR axis
        nifti_ornt = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(nifti.affine))
        lr_axis = int(nifti_ornt[0, 0])
        # Flip and store the image
        img_flipped = np.flip(nifti.get_fdata(), axis=lr_axis)
        img_flipped_fpath = os.path.join(os.path.dirname(out_fpath), f'flipped_{n}.nii.gz')
        nib.Nifti1Image(img_flipped, nifti.affine, nifti.header ).to_filename(img_flipped_fpath)

        if n == 0:
            # Register within itself
            register_affine_sitk(
                ref=in_fpath,
                mov=img_flipped_fpath,
                transform_fpath=sym_tx_fpath,
                warped_fpath=out_fpath,
                warp_interpolation=sitk.sitkLanczosWindowedSinc,
                num_threads=os.cpu_count())
        else:
            transform_sitk_image(
                mov=img_flipped_fpath,
                ref=in_fpath,
                transform=sym_tx_fpath,
                warped_fpath=out_fpath,
                interpolator=sitk.sitkLanczosWindowedSinc)

        os.remove(img_flipped_fpath)



if __name__ == '__main__':
    compute_symmetric_modalities(
        in_fpaths=[
            '/home/albert/Desktop/ISLES22/datasets/prepared/sub-strokecase0001_ses-0001/dwi_ref.nii.gz',
            '/home/albert/Desktop/ISLES22/datasets/prepared/sub-strokecase0001_ses-0001/adc_ref.nii.gz',
        ],
        out_fpaths=[
            '/home/albert/Desktop/ISLES22/datasets/prepared/sub-strokecase0001_ses-0001/sym_dwi_ref.nii.gz',
            '/home/albert/Desktop/ISLES22/datasets/prepared/sub-strokecase0001_ses-0001/sym_adc_ref.nii.gz',
            ]
    )