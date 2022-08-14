import logging
import os
import shutil
import tempfile

from .utils import run_bash
from .path import get_path, remove_ext, get_filename, make_dirs

import numpy as np
import nibabel as nib



def perform_halfway_registration(
        baseline_fpath,
        followup_fpath,
        baseline_half_fpath,
        followup_half_fpath,
        baseline2half_fpath,
        followup2half_fpath,
        average_half_fpath=None,
        interp='LanczosWindowedSinc',
        decimals=2
    ):

    # Get dimensions names and paths

    tempdir = tempfile.TemporaryDirectory()
    tmp_path = tempdir.name

    baseline_temp_fpath = os.path.join(tmp_path, get_filename(baseline_fpath))
    shutil.copy(baseline_fpath, baseline_temp_fpath)

    followup_temp_fpath = os.path.join(tmp_path, get_filename(followup_fpath))
    shutil.copy(followup_fpath, followup_temp_fpath)

    # Declare names for intermediate stages
    B, F = baseline_temp_fpath, followup_temp_fpath
    ndims = nib.load(B).header['dim'][0]
    bname = get_filename(B, ext=False)
    fname = get_filename(F, ext=False)
    Binit = os.path.join(tmp_path, f'{bname}_init_half')
    Finit = os.path.join(tmp_path, f'{fname}_init_half')
    Bhalf = os.path.join(tmp_path, f'{bname}_half')
    Fhalf = os.path.join(tmp_path, f'{fname}_half')

    # Generate the filepaths that will be output from ANTS functions
    Binit_id_fp = f'{Binit}_id.mat'
    Finit_inv_fp = f'{Finit}_inv.mat'
    Binit2half_fp = f'{Binit}0GenericAffine.mat'
    Finit2half_fp = f'{Finit}0GenericAffine.mat'

    half_avg_tx_fp = os.path.join(tmp_path, f'half_avg.mat')
    half_avg_fp = os.path.join(tmp_path, f'half.nii.gz')
    half_mid_fp = os.path.join(tmp_path, f'half_mid.mat')

    Bhalf_fp = f'{Bhalf}_aff.nii.gz'
    Fhalf_fp = f'{Fhalf}_aff.nii.gz'
    B2half_fp = f'{Bhalf}0GenericAffine.mat'
    F2half_fp = f'{Fhalf}0GenericAffine.mat'

    # Predeclare useful comand parts
    reg = f'antsRegistration -d {ndims}'
    aff = ' -t affine[ 0.25 ]  -c [ 1009x200x20,1.e-8,20 ]  -s 4x2x0 -f 4x2x1 '

    if any([not os.path.isfile(fp) for fp in (baseline2half_fpath, followup2half_fpath, half_avg_fp)]):
        print('Running registration...')
        ### Register in both directions, then average the result
        run_bash(
            f'{reg} -r [ {B}, {F}, 1 ] -m mattes[ {B}, {F}, 1 , 32 , regular , 0.25 ] {aff} -z 1 -o [ {Binit} ]', v=False)
        run_bash(
            f'{reg} -r [ {F}, {B}, 1 ] -m mattes[ {F}, {B}, 1 , 32 , regular , 0.25 ] {aff} -z 1 -o [ {Finit} ]', v=False)
        # get the identity map
        run_bash(
            f'ComposeMultiTransform {ndims} {Binit_id_fp} -R {Binit2half_fp}  {Binit2half_fp} -i {Binit2half_fp}', v=False)
        # invert the 2nd affine registration map
        run_bash(f'ComposeMultiTransform {ndims} {Finit_inv_fp} -R {Binit2half_fp} -i {Finit2half_fp}', v=False)
        # get the average affine map
        run_bash(f'AverageAffineTransform {ndims} {half_avg_tx_fp}  {Finit_inv_fp} {Binit2half_fp}', v=False)
        # get the midpoint affine map
        run_bash(f'AverageAffineTransform {ndims} {half_mid_fp} {Binit_id_fp}  {half_avg_tx_fp}', v=False)

        # this applies, to F, B map from F to midpoint(F,B)
        run_bash(f'antsApplyTransforms -d {ndims} -i {F} -o {half_avg_fp} -t {half_mid_fp} -r {B}', v=False)

        # compute the map from B to midpoint(F,B)
        run_bash(f'{reg} -r  [ {half_avg_fp}, {B}, 1 ] -m mattes[ {half_avg_fp}, {B}, 1 , 32, random , 0.25 ] ' +
                 f'{aff} -n {interp} -o [ {Bhalf}, {Bhalf_fp}]', v=False)
        # compute the map from F to midpoint(F,B)
        run_bash(f'{reg} -r [ {Bhalf_fp}, {F}, 1 ] -m mattes[ {Bhalf_fp}, {F}, 1 , 32, random , 0.25 ] ' +
                 f'{aff} -n {interp} -o [ {Fhalf}, {Fhalf_fp}]', v=False)

        # Copy transforms to output directories
        shutil.copy(B2half_fp, baseline2half_fpath)
        shutil.copy(F2half_fp, followup2half_fpath)

        # Remove all temporary files
        to_remove = [half_avg_tx_fp, half_mid_fp, Binit_id_fp, Finit_inv_fp, Binit2half_fp, Finit2half_fp,
                     B2half_fp, F2half_fp]
        [os.remove(fp) for fp in to_remove]
    else:
        print('Found transforms, applying...')
        tx_cmd = f'antsApplyTransforms -d {ndims} -n {interp}'
        run_bash(f'{tx_cmd} -i {B} -o {Bhalf_fp} -t {baseline2half_fpath} -r {half_avg_fp}', v=False)
        run_bash(f'{tx_cmd} -i {F} -o {Fhalf_fp} -t {followup2half_fpath} -r {half_avg_fp}', v=False)

    # Postprocess output images in all cases
    Bhalf_nifti, Fhalf_nifti = nib.load(Bhalf_fp), nib.load(Fhalf_fp)
    Bhalf_out= np.round(np.clip(Bhalf_nifti.get_fdata(), 0.0, None), decimals=decimals).astype(float)
    Fhalf_out= np.round(np.clip(Fhalf_nifti.get_fdata(), 0.0, None), decimals=decimals).astype(float)
    nib.Nifti1Image(Bhalf_out, Bhalf_nifti.affine, Bhalf_nifti.header).to_filename(baseline_half_fpath)
    nib.Nifti1Image(Fhalf_out, Fhalf_nifti.affine, Fhalf_nifti.header).to_filename(followup_half_fpath)

    # Cleanup the temporary directory
    tempdir.cleanup()



def register_halfway(
        baseline_fpath,
        followup_fpath,
        baseline_half_fpath=None,
        followup_half_fpath=None,
        baseline2half_fpath=None,
        followup2half_fpath=None,
        average_half_fpath=None,
        interp='LanczosWindowedSinc',
        decimals=2
    ):

    # Prepare temporary directory for registration
    tempdir = tempfile.TemporaryDirectory()

    # Copy input images to this directory
    baseline_temp_fpath = os.path.join(tempdir.name, get_filename(baseline_fpath))
    followup_temp_fpath = os.path.join(tempdir.name, get_filename(followup_fpath))
    shutil.copy(baseline_fpath, baseline_temp_fpath)
    shutil.copy(followup_fpath, followup_temp_fpath)

    # Declare names for intermediate stages
    B, F = baseline_temp_fpath, followup_temp_fpath
    ndims = nib.load(B).header['dim'][0]
    bname = get_filename(B, ext=False)
    fname = get_filename(F, ext=False)
    Binit = os.path.join(tempdir.name, f'{bname}_init_half')
    Finit = os.path.join(tempdir.name, f'{fname}_init_half')
    Bhalf = os.path.join(tempdir.name, f'{bname}_half')
    Fhalf = os.path.join(tempdir.name, f'{fname}_half')

    # Generate the filepaths that will be output from ANTS functions
    Binit_id_fp = f'{Binit}_id.mat'
    Finit_inv_fp = f'{Finit}_inv.mat'
    Binit2F_fp = f'{Binit}0GenericAffine.mat'
    Finit2B_fp = f'{Finit}0GenericAffine.mat'

    half_tx_fp = os.path.join(tempdir.name, f'half_tx.mat')
    half_img_fp = os.path.join(tempdir.name, f'half.nii.gz')
    F2half_tx_fp = os.path.join(tempdir.name, f'half_mid.mat')

    Bhalf_fp = f'{Bhalf}_aff.nii.gz'
    Fhalf_fp = f'{Fhalf}_aff.nii.gz'
    B2half_fp = f'{Bhalf}0GenericAffine.mat'
    F2half_fp = f'{Fhalf}0GenericAffine.mat'

    # Predeclare useful comand parts
    reg = f'antsRegistration -d {ndims}'
    aff = ' -t affine[ 0.25 ]  -c [ 1009x200x20,1.e-8,20 ]  -s 4x2x0 -f 4x2x1 '

    logging.info('Running registration...')
    ### Register in both directions, then average the result
    run_bash(
        f'{reg} -r [ {B}, {F}, 1 ] -m mattes[ {B}, {F}, 1 , 32 , regular , 0.25 ] {aff} -z 1 -o [ {Binit} ]', v=False)
    run_bash(
        f'{reg} -r [ {F}, {B}, 1 ] -m mattes[ {F}, {B}, 1 , 32 , regular , 0.25 ] {aff} -z 1 -o [ {Finit} ]', v=False)

    # get the identity map -> B2B
    run_bash(
        f'ComposeMultiTransform {ndims} {Binit_id_fp} -R {Binit2F_fp} {Binit2F_fp} -i {Binit2F_fp}', v=False)
    # Get inverse mapping -> B2F
    run_bash(f'ComposeMultiTransform {ndims} {Finit_inv_fp} -R {Binit2F_fp} -i {Finit2B_fp}', v=False)
    # get the average affine map
    run_bash(f'AverageAffineTransform {ndims} {half_tx_fp} {Finit_inv_fp} {Binit2F_fp}', v=False)
    # get the midpoint affine map
    run_bash(f'AverageAffineTransform {ndims} {F2half_tx_fp} {Binit_id_fp} {half_tx_fp}', v=False)


    # this applies, to F, B map from F to midpoint(F,B)
    run_bash(f'antsApplyTransforms -d {ndims} -n {interp} -i {F} -o {half_img_fp} -t {F2half_tx_fp} -r {B}', v=False)

    # compute the map from B to midpoint(F,B)
    run_bash(f'{reg} -r  [ {half_img_fp}, {B}, 1 ] -m mattes[ {half_img_fp}, {B}, 1 , 32, random , 0.25 ] ' +
             f'{aff} -n {interp} -o [ {Bhalf}, {Bhalf_fp}]', v=False)

    # compute the map from F to midpoint(F,B)
    run_bash(f'{reg} -r [ {Bhalf_fp}, {F}, 1 ] -m mattes[ {Bhalf_fp}, {F}, 1 , 32, random , 0.25 ] ' +
             f'{aff} -n {interp} -o [ {Fhalf}, {Fhalf_fp}]', v=False)

    # Obtain intensity ranges of input image for clipping to original range
    # (since LanczosSinc can return values outside of the original intensity range)
    baseline_array = nib.load(baseline_temp_fpath).get_fdata()
    followup_array = nib.load(followup_temp_fpath).get_fdata()
    base_range = (np.min(baseline_array), np.max(baseline_array))
    fwup_range = (np.min(followup_array), np.max(followup_array))
    half_range = (np.min((base_range[0], fwup_range[0])), np.max((base_range[1], fwup_range[1])))

    if baseline_half_fpath is not None:
        os.makedirs(os.path.dirname(baseline_half_fpath), exist_ok=True)
        Bhalf_nifti = nib.load(Bhalf_fp)
        Bhalf_out = \
            np.round(np.clip(Bhalf_nifti.get_fdata(), base_range[0], base_range[1]), decimals=decimals).astype(float)
        nib.Nifti1Image(Bhalf_out, Bhalf_nifti.affine, Bhalf_nifti.header).to_filename(baseline_half_fpath)

    if followup_half_fpath is not None:
        os.makedirs(os.path.dirname(followup_half_fpath), exist_ok=True)
        Fhalf_nifti = nib.load(Fhalf_fp)
        Fhalf_out = \
            np.round(np.clip(Fhalf_nifti.get_fdata(), fwup_range[0], fwup_range[1]), decimals=decimals).astype(float)
        nib.Nifti1Image(Fhalf_out, Fhalf_nifti.affine, Fhalf_nifti.header).to_filename(followup_half_fpath)

    if baseline2half_fpath is not None:
        os.makedirs(os.path.dirname(baseline2half_fpath), exist_ok=True)
        shutil.copy(B2half_fp, baseline2half_fpath)

    if followup2half_fpath is not None:
        os.makedirs(os.path.dirname(followup2half_fpath), exist_ok=True)
        shutil.copy(F2half_fp, followup2half_fpath)

    if average_half_fpath is not None:
        os.makedirs(os.path.dirname(average_half_fpath), exist_ok=True)
        avg_half_nifti = nib.load(half_img_fp)
        avg_half_out = \
            np.round(np.clip(avg_half_nifti.get_fdata(), half_range[0], half_range[1]), decimals=decimals).astype(float)
        nib.Nifti1Image(avg_half_out, avg_half_nifti.affine, avg_half_nifti.header).to_filename(average_half_fpath)

    # Cleanup the temporary directory
    tempdir.cleanup()


def apply_linear_transform(in_fp, ref_fp, tx_fp, out_fp, interp='Linear', out_value=0, out_dtype='float', ndims=3):
    apply_tx_cmd = f'antsApplyTransforms -d {ndims} -i {interp} -f {out_value} -u {out_dtype}'
    run_bash(f'{apply_tx_cmd} -i {in_fp} -o {out_fp} -t {tx_fp} -r {ref_fp}', v=False)


def perform_nonlinear_registration(
        baseline_fpath,
        followup_fpath,
        baseline_initial_tx,
        followup_initial_tx,
        image_out_fpath=None,
        fields_out_fpath=None,
        fields_inv_out_fpath=None,
        ndims=3):

    # Prepare temporary directory for registration and copy all input files there
    tempdir = tempfile.TemporaryDirectory()
    tmp_output_pathname = os.path.join(tempdir.name, 'nonlin_')

    reg = f'antsRegistration -d {ndims}'
    initial_transforms = \
        f'--initial-fixed-transform {baseline_initial_tx} --initial-moving-transform {followup_initial_tx}'
    output = f'-o [{tmp_output_pathname}, {tmp_output_pathname}image.nii.gz]'

    ########################### antsRegistrationSyNQuick.sh
    # metric = f'-m MI[ {baseline_fpath}, {followup_fpath} , 1 , 32 ]'
    # transform = f'-t syn[ 0.1, 3, 0.0 ] -c [ 100x70x50x0,1e-6,10 ] -f 8x4x2x1 -s 3x2x1x0vox'

    ########################### antsRegistrationSyN.sh
    # print('    Using params from antsRegistrationSyN.sh')
    # metric = f'-m CC[ {baseline_fpath}, {followup_fpath} , 1 , 4 ]'
    # transform = f'-t syn[ 0.1, 3, 0.0 ] -c [ 100x70x50x20,1e-6,10 ] -f 8x4x2x1 -s 3x2x1x0vox'

    ########################### SyNOnly
    # print('    Using params from SyNOnly')
    # metric = f'-m mattes[{baseline_fpath}, {followup_fpath}, 1, 32]'
    # transform = f'-t SyN[0.2, 3, 0] -c [40x20x0,1e-7,8] -f 4x2x1 -s 2x1x0 -u 1'

    ########################### SyNCC
    print('    Using params from SyNCC')
    metric = f'-m CC[{baseline_fpath}, {followup_fpath}, 1, 4]'
    #transform = f'-t SyN[0.15,3,0] -c [100x70x50x20,1e-7,8] -f 4x3x2x1 -s 3x2x1x0 -u 1' # 0.02 %

    # print('    Using params from SyNCC half iterations')
    # transform = f'-t SyN[0.15,3,0] -c [50x35x25x10,1e-7,8] -f 4x3x2x1 -s 3x2x1x0 -u 1'

    print('    Using params from SyNCC aprox third iterations')
    transform = f'-t SyN[0.15,3,0] -c [40x20x10x5,1e-7,8] -f 4x3x2x1 -s 3x2x1x0 -u 1'

    run_bash(cmd=f'{reg} {initial_transforms} {metric} {transform} {output}', v=False)

    # Copy results to output paths
    fields_ants_fpath = f'{tmp_output_pathname}1Warp.nii.gz'
    if fields_out_fpath is not None:
        os.makedirs(os.path.dirname(fields_out_fpath), exist_ok=True)
        shutil.copy(fields_ants_fpath, fields_out_fpath)

    fields_inv_ants_fpath = f'{tmp_output_pathname}1InverseWarp.nii.gz'
    if fields_inv_out_fpath is not None:
        os.makedirs(os.path.dirname(fields_inv_out_fpath), exist_ok=True)
        shutil.copy(fields_inv_ants_fpath, fields_inv_out_fpath)

    image_ants_fpath = f'{tmp_output_pathname}image.nii.gz'
    if image_out_fpath is not None:
        os.makedirs(os.path.dirname(image_out_fpath), exist_ok=True)
        shutil.copy(image_ants_fpath, image_out_fpath)

    # Cleanup the temporary directory
    tempdir.cleanup()


def compute_jacobian(fields_fp, jacobian_out_fp, ndims=3, do_logjacobian=0, use_geometric=1):
    run_bash(
        f'CreateJacobianDeterminantImage {ndims} {fields_fp} {jacobian_out_fp} {do_logjacobian} {use_geometric}', v=False)