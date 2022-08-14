import os
import shutil
import SimpleITK as sitk
import tempfile
import nibabel as nib
import numpy as np
import copy
import torch
from tqdm import tqdm

from utils_simpleitk import register_affine_sitk, transform_sitk_image, change_spacing_sitk_image, resample_spacing_sitk_image
from utils_inference import compute_symmetric_modalities

from datamodule_isles22 import ISLES22_DataModule
from module_isles22_unet import ISLES22_UnetModule
from model_sunet_varx import SUNETx4_varX
from monai.losses import FocalLoss, DiceCELoss

def image_inference_isles22_UnetEnsembleAverage(
        image,
        nifti,
        device,
        probs_fpath,
        models_filepaths
    ):

    PREPARED_ROOT = '/'
    dice_ce_func = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_ce_loss = lambda out, tgt: dice_ce_func(out, tgt[:, 0:1, :, :, :].long())
    cfg = {
            'seed': 1,
            'gpus': [0],

            'prepared_path': PREPARED_ROOT + '_1x1x2',
            'input_modalities': ('flair', 'adc', 'dwi'),

            'patch_type': 'isotropic',
            'patch_shape': (24, 24, 24),

            'num_patches': 250 * 1000,
            'min_patches_lesion': 10,
            'sampling_fractions': (0.01, 0.49, 0.50),
            'batch_size': 2,

            'norm_type': 'percentile_minmax',
            'do_symmetric_modalities': True,
            'augmentation_prob': 0.0,

            # Crossvalidation
            'crossvalidation_type': 'all', # ('all', 'single_scanner')
            'num_folds': 5,

            # Training
            'do_train': True,

            'loss': dice_ce_loss,
            'max_epochs': 100,
            'train_fraction': 0.8,
            'batch_size': 16,
            'patience': 5,

            'optimizer': 'Adadelta', # one of ('Adam', 'Adadelta')
            'optimizer_opts': {},
            'optimizer_lr': 0.05,
            'auto_lr_find': False,
            
            # Inference
            'do_inference': True,
            #'extraction_step': (24, 24, 24)
            'extraction_step': (9, 9, 9)
        }

    optimizer_dict = {
        'Adadelta': torch.optim.Adadelta,
        'Adam': torch.optim.Adam}

    num_input_channels = len(cfg['input_modalities'])
    if cfg['do_symmetric_modalities']:
        num_input_channels *= 2


    datamodule = ISLES22_DataModule(
        isles22_original_dict={},
        prepared_path=cfg['prepared_path'],
        template_fpath=None,
        num_folds=cfg['num_folds'],
        input_modalities=cfg['input_modalities'],
        do_symmetric_modalities=cfg['do_symmetric_modalities'],
        patch_type=cfg['patch_type'],
        patch_shape=cfg['patch_shape'],
        num_patches=cfg['num_patches'],
        sampling_fractions=cfg['sampling_fractions'],
        augmentation_prob=cfg['augmentation_prob'],
        norm_type=cfg['norm_type'],
        batch_size=cfg['batch_size'],
        extraction_step=cfg['extraction_step'],
        num_threads=os.cpu_count(),
        max_cases=None,
        min_patches_per_lesion=cfg['min_patches_lesion']
    )

    ensemble_models =  [
        ISLES22_UnetModule.load_from_checkpoint(ckpt_fold_fpath,
            unet_model=SUNETx4_varX(
                in_ch=num_input_channels, 
                out_ch=2,
            ),
            activation=torch.nn.Softmax(dim=1),
            segmentation_loss=cfg['loss'],
            optimizer_class=optimizer_dict[cfg['optimizer']],
            optimizer_lr=cfg['optimizer_lr'],
            optimizer_opts=cfg['optimizer_opts'])
        for ckpt_fold_fpath in models_filepaths]

    if probs_fpath is not None:
        if os.path.isfile(probs_fpath):
            return

    input_shape = image.shape
    image = copy.deepcopy(image)

    # First pad image to ensure all voxels in volume are processed independently of extraction_step
    pad_dims = [(int(0),)] + [(int(np.ceil(patch_dim / 2.0)),) for patch_dim in datamodule.patch_shape]
    # Also generate unpad_slice to undo the padding at the end
    unpad_slice = tuple([slice(None)] + [slice(pad_dim[0], img_dim + pad_dim[0]) 
                                        for pad_dim, img_dim in zip(pad_dims[1:], image.shape[1:])])

    image = np.pad(image, pad_dims, mode='edge')
    inference_dataloader = datamodule.image_inference_dataloader(image)

    # Perform patch inference
    accum_img, count_img = None, None

    [em.eval() for em in ensemble_models]
    [em.to(device) for em in ensemble_models]
    with torch.no_grad():
        for x, x_slices in tqdm(inference_dataloader):
            x = x.to(device)
            output = torch.mean(torch.stack([em(x) for em in ensemble_models], dim=0), dim=0)

            if accum_img is None or count_img is None:
                accum_img = output.new_zeros((output.shape[1],) + image.shape[1:])
                count_img = torch.zeros_like(accum_img)
                
            for output_patch, patch_slice in zip(output, x_slices):
                accum_img[patch_slice] += output_patch
                count_img[patch_slice] += torch.ones_like(output_patch)

    # Perform overlapping prediction averaging
    count_img[count_img == 0] = 1
    infered_probs = torch.div(accum_img, count_img).detach().cpu().numpy()
    # Unpad the image
    infered_probs = infered_probs[unpad_slice]
    assert np.array_equal(input_shape[1:], infered_probs.shape[1:]), \
        f'{input_shape[1:]} != {infered_probs.shape[1:]}'
    
    if probs_fpath is not None:
        os.makedirs(os.path.dirname(probs_fpath), exist_ok=True)

        nib.Nifti1Image(
            np.round(infered_probs[1], decimals=4), nifti.affine, nifti.header
        ).to_filename(probs_fpath)

    return infered_probs


def segment_isles22_pipeline(
    dwi_sitk_img,
    adc_sitk_img,
    flair_sitk_img,
    models_filepaths,
    threshold=0.5,
    probs_ref_fpath_out=None
):
    tempdir = tempfile.TemporaryDirectory()
    case_path_out = tempdir.name

    dwi_prep_fpath = os.path.join(case_path_out, 'dwi_ref.nii.gz')
    adc_prep_fpath = os.path.join(case_path_out, 'adc_ref.nii.gz')
    flair_prep_fpath = os.path.join(case_path_out, 'flair_ref.nii.gz')

    ### 1. PREPROCESSING
    
    # Store original image to have nifti reference headers
    dwi_orig_img = copy.deepcopy(dwi_sitk_img)

    # Correct zero values
    dwi_sitk_img = sitk.Clamp(dwi_sitk_img, outputPixelType=sitk.sitkFloat32, lowerBound=0.0)
    adc_sitk_img = sitk.Clamp(adc_sitk_img, outputPixelType=sitk.sitkFloat32, lowerBound=0.0)
    flair_sitk_img = sitk.Clamp(flair_sitk_img, outputPixelType=sitk.sitkFloat32, lowerBound=0.0)

    # Resample diffusion images to 1x1x2
    print('Resampling...')
    change_spacing_sitk_image(
            img=dwi_sitk_img,
            out_spacing=(1.0, 1.0, 2.0),
            interpolator=sitk.sitkLanczosWindowedSinc,
            warped_fpath=dwi_prep_fpath)

    change_spacing_sitk_image(
        img=adc_sitk_img,
        out_spacing=(1.0, 1.0, 2.0),
        interpolator=sitk.sitkLanczosWindowedSinc,
        warped_fpath=adc_prep_fpath)

    # Register FLAIR to dwi
    print('Registering FLAIR...')
    register_affine_sitk(
        ref=dwi_prep_fpath,
        mov=flair_sitk_img,
        transform_fpath=os.path.join(case_path_out, 'flair_to_dwi.mat'),
        warp_interpolation=sitk.sitkLanczosWindowedSinc,
        warped_fpath=flair_prep_fpath,
        num_threads=os.cpu_count())

    # Compute symmetric modalities
    mod_fpaths = [flair_prep_fpath, adc_prep_fpath, dwi_prep_fpath]
    sym_mod_fpaths = [
        os.path.join(case_path_out, 'sym_flair_ref.nii.gz'),
        os.path.join(case_path_out, 'sym_adc_ref.nii.gz'),
        os.path.join(case_path_out, 'sym_dwi_ref.nii.gz')
    ]

    if not all([os.path.isfile(smfp) for smfp in sym_mod_fpaths]):
        print('Computing symmetric modalities')
        compute_symmetric_modalities(in_fpaths=mod_fpaths, out_fpaths=sym_mod_fpaths)

    ### 2. INFERENCE with ENSEMBLE
    probs_ref_fpath = os.path.join(case_path_out, 'probs_ref.nii.gz')
    mask_ref_fpath = os.path.join(case_path_out, 'mask_ref.nii.gz')
    mask_orig_fpath = os.path.join(case_path_out, 'mask_orig.nii.gz')

    # Load image array
    images_fpaths = mod_fpaths + sym_mod_fpaths
    image_niftis = [nib.load(ifp) for ifp in images_fpaths]
    image_array = np.stack([image_nifti.get_fdata() for image_nifti in image_niftis], axis=0).astype(np.float16)
    [image_nifti.uncache() for image_nifti in image_niftis]

    print('Inference with ensemble (might take long)...')
    image_inference_isles22_UnetEnsembleAverage(
        image=image_array,
        nifti=image_niftis[0],
        device='cuda',
        probs_fpath=probs_ref_fpath,
        models_filepaths=models_filepaths
    )

    ### 3. Resample back to original spacing
    if probs_ref_fpath_out is not None:
        shutil.copy(probs_ref_fpath, probs_ref_fpath_out)

    # Binarize probs
    ref_probs_nifti = nib.load(probs_ref_fpath)
    nib.Nifti1Image(
        (ref_probs_nifti.get_fdata() > threshold).astype(np.uint), 
        ref_probs_nifti.affine, 
        ref_probs_nifti.header
    ).to_filename(mask_ref_fpath)

    mask_orig_sitk = resample_spacing_sitk_image(
        mov=mask_ref_fpath,
        ref=dwi_orig_img,
        interpolator=sitk.sitkLabelGaussian,
        warped_fpath=mask_orig_fpath)

    mask_out = (sitk.GetArrayFromImage(mask_orig_sitk) > 0.5).astype(int)

    tempdir.cleanup()

    return mask_out