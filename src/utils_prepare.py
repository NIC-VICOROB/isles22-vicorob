import copy
import os
import shutil
import nibabel as nib
import numpy as np
import acglib as acg
from tqdm import tqdm

import SimpleITK as sitk
from utils_simpleitk import register_affine_sitk, transform_sitk_image, change_spacing_sitk_image, resample_spacing_sitk_image
from utils_inference import compute_symmetric_modalities

def crop_borders(image, background_value=0):
    """Crops the background borders of an image. If any given channel is all background, it will also be cropped!
    :param image: the image to crop.
    :param background_value: value of the background that will be cropped.
    :return: The image with background borders cropped.
    """
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
    return image[tuple(crop_slice)]


def prepare_isles22_resample_case(case_dict, case_path_out, template_fpath):    
    os.makedirs(case_path_out, exist_ok=True)

    ### 1. Copy IMAGES and correct ZERO VALUE
    flair_fpath = os.path.join(case_path_out, 'flair.nii.gz')

    if not os.path.isfile(flair_fpath):
        nib.load(case_dict['flair']['fpath']).to_filename(flair_fpath)

    adc_fpath = os.path.join(case_path_out, 'adc.nii.gz')
    if not os.path.isfile(adc_fpath):
        adc_nifti = nib.load(case_dict['adc']['fpath'])
        nib.Nifti1Image(
            np.clip(adc_nifti.get_fdata(), 0.0, None), adc_nifti.affine, adc_nifti.header
        ).to_filename(adc_fpath)

    dwi_fpath = os.path.join(case_path_out, 'dwi.nii.gz')
    if not os.path.isfile(dwi_fpath):
        dwi_nifti = nib.load(case_dict['dwi']['fpath'])
        nib.Nifti1Image(
            np.clip(dwi_nifti.get_fdata(), 0.0, None), dwi_nifti.affine, dwi_nifti.header
        ).to_filename(dwi_fpath)

    msk_fpath = os.path.join(case_path_out, 'msk.nii.gz')
    if not os.path.isfile(msk_fpath):    
        nib.load(case_dict['msk']['fpath']).to_filename(msk_fpath)

    ### RESAMPLING to 1x1x2
    dwi_in_ref_fpath = os.path.join(case_path_out, 'dwi_ref.nii.gz')
    if not os.path.isfile(dwi_in_ref_fpath):
        change_spacing_sitk_image(
            img=dwi_fpath,
            out_spacing=(1.0, 1.0, 2.0),
            warped_fpath=dwi_in_ref_fpath,
            interpolator=sitk.sitkLanczosWindowedSinc)


    adc_in_ref_fpath = os.path.join(case_path_out, 'adc_ref.nii.gz')
    if not os.path.isfile(adc_in_ref_fpath):
        change_spacing_sitk_image(
            img=adc_fpath,
            out_spacing=(1.0, 1.0, 2.0),
            warped_fpath=adc_in_ref_fpath,
            interpolator=sitk.sitkLanczosWindowedSinc)

    msk_in_ref_fpath = os.path.join(case_path_out, 'msk_ref.nii.gz')
    if not os.path.isfile(msk_in_ref_fpath):
        change_spacing_sitk_image(
            img=msk_fpath,
            out_spacing=(1.0, 1.0, 2.0),
            warped_fpath=msk_in_ref_fpath,
            interpolator=sitk.sitkNearestNeighbor)

    flair_in_ref_fpath = os.path.join(case_path_out, 'flair_ref.nii.gz')
    if not os.path.isfile(flair_in_ref_fpath):
        register_affine_sitk(
            ref=dwi_in_ref_fpath,
            mov=flair_fpath,
            transform_fpath=os.path.join(case_path_out, 'flair_to_dwi.mat'),
            warped_fpath=flair_in_ref_fpath,
            warp_interpolation=sitk.sitkLanczosWindowedSinc,
            num_threads=max(1, os.cpu_count() - 3)
        )
        
        # resample_spacing_sitk_image(
        #     mov=flair_fpath,
        #     ref=dwi_in_ref_fpath,
        #     interpolator=sitk.sitkLanczosWindowedSinc,
        #     warped_fpath=flair_in_ref_fpath)


    ### GENERATE SYMMETRIC MODALITIES
    mod_fpaths = [flair_in_ref_fpath, adc_in_ref_fpath, dwi_in_ref_fpath]
    sym_mod_fpaths = [
        os.path.join(case_path_out, 'sym_flair_ref.nii.gz'),
        os.path.join(case_path_out, 'sym_adc_ref.nii.gz'),
        os.path.join(case_path_out, 'sym_dwi_ref.nii.gz')
    ]

    if not all([os.path.isfile(smfp) for smfp in sym_mod_fpaths]):
        compute_symmetric_modalities(in_fpaths=mod_fpaths, out_fpaths=sym_mod_fpaths)


def load_prepared_isles22_data(num_threads, prepared_path, modalities=('flair', 'adc', 'dwi'), load_images=True, max_cases=None):
    def load_prepared_case(case_entry):
        # Generate filepaths
        images_fpaths = [os.path.join(case_entry.path, f'{img_name}_ref.nii.gz') for img_name in modalities]
        msk_fpath = os.path.join(case_entry.path, f'msk_ref.nii.gz')

        # Load niftis
        image_niftis = [nib.load(ifp) for ifp in images_fpaths]
        msk_nifti = nib.load(msk_fpath)

        prep_case = {
            'path': case_entry.path,
            'msk_orig_fpath': os.path.join(case_entry.path, f'msk.nii.gz'),
            'images_fpaths': images_fpaths,
            'msk_fpath': msk_fpath,
            'nifti_original': nib.load(os.path.join(case_entry.path, 'msk.nii.gz')),
            'nifti_ref': msk_nifti,
            'flair_to_ref_fpath': os.path.join(case_entry.path, 'flair_to_ref.mat'),
            'diffusion_to_flair_fpath': os.path.join(case_entry.path, 'diffusion_to_flair.mat')
        }

        if load_images:
            # Load arrays and format to 4-dims
            image_array = np.stack([image_nifti.get_fdata() for image_nifti in image_niftis], axis=0).astype(np.float16)
            msk_array = np.expand_dims(msk_nifti.get_fdata(), axis=0).astype(np.uint8)
            
            # Uncache nibabel's nifti arrays (we derived images)
            [image_nifti.uncache() for image_nifti in image_niftis]
            msk_nifti.uncache()

            prep_case.update({
                'image': image_array,
                'msk': msk_array,
            })
        
        return prep_case
    
    case_entries = acg.list_dirs(prepared_path)

    if max_cases is not None:
        case_entries = case_entries[:max_cases]

    prepared_data_list = acg.parallel_run(
        func=load_prepared_case,
        args=[[ce] for ce in case_entries],
        num_threads=num_threads,
        do_print_progress_bar=True,
        progress_bar_prefix='Loading prepared data')

    return {ce.name: cdict for ce, cdict in zip(case_entries, prepared_data_list)}

