import subprocess
import os
import shutil
import nibabel as nib
import numpy as np
import tempfile

from .path import remove_ext, make_dirs, get_path, get_filename


def run_fsl_anat(fpath_in, do_fast, do_first, merge_fast_first, remove_anat_dir=True):
    assert os.path.isfile(fpath_in), fpath_in

    anat_dir = remove_ext(fpath_in) + '.anat'
    fpaths_anat_fast = [os.path.join(anat_dir, 'T1_fast_pve_{}.nii.gz'.format(i)) for i in range(3)]
    fpath_anat_first = os.path.join(anat_dir, 'T1_subcort_seg.nii.gz')

    fpaths_tgt_fast = [remove_ext(fpath_in) + '_fast_{}.nii.gz'.format(i) for i in range(4)]
    fpaths_tgt_fast_first = [remove_ext(fpath_in) + '_fast_first_{}.nii.gz'.format(i) for i in range(4)]

    if not os.path.isfile(fpaths_tgt_fast_first[-1]):
        if not all([os.path.isfile(faf) for faf in fpaths_anat_fast]) or not os.path.isfile(fpath_anat_first):
            #anat_cmd =  f'fsl_anat --clobber --weakbias --nocrop --noreorient -i {fpath_in}'
            anat_cmd =  f'fsl_anat --weakbias --nocrop --noreorient -i {fpath_in}'
            print(anat_cmd)
            subprocess.check_output(['bash', '-c', anat_cmd])

        # Load FAST segmentations, add background prob channel and store in the target folder
        fast_nifti = nib.load(fpaths_anat_fast[0])
        fast_pves = [nib.load(fp).get_data() for fp in fpaths_anat_fast]
        fast_pves.insert(0, 1.0 - np.sum(np.stack(fast_pves, axis=0), axis=0))  # Background probability
        for fast_pve, fpath_tgt_fast in zip(fast_pves, fpaths_tgt_fast):
            fast_pve_arr = np.round(fast_pve, decimals=5)
            nib.Nifti1Image(fast_pve_arr, fast_nifti.affine, fast_nifti.header).to_filename(fpath_tgt_fast)

        # Load first segmentation, overwrite the subcortical structures and store in target folder
        first_seg = nib.load(fpath_anat_first).get_data()
        fast_pves[1][first_seg > 0] = 0.0
        fast_pves[2][first_seg > 0] = 1.0
        fast_pves[3][first_seg > 0] = 0.0
        for fast_pve, fpath_tgt_fast_first in zip(fast_pves, fpaths_tgt_fast_first):
            fast_pve_arr = np.round(fast_pve, decimals=5)
            nib.Nifti1Image(fast_pve_arr, fast_nifti.affine, fast_nifti.header).to_filename(fpath_tgt_fast_first)

        if remove_anat_dir:
            shutil.rmtree(anat_dir)


def run_fast(filepath_in):
    print('Running FAST: {}'.format(filepath_in))
    subprocess.check_call(['bash', '-c', 'fast {}'.format(filepath_in)])

    pve_fpaths = [remove_ext(filepath_in) + '_pve_{}.nii.gz'.format(i) for i in range(3)]
    out_fpaths = [remove_ext(filepath_in) + '_fast_{}.nii.gz'.format(i) for i in range(3)]

    for pve_fpath, out_fpath in zip(pve_fpaths, out_fpaths):
        os.rename(pve_fpath, out_fpath)

    # Remove all other files
    os.remove(os.path.join(remove_ext(filepath_in) + '_mixeltype.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_pveseg.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_seg.nii.gz'))



def run_first(filepath_in, is_skull_stripped=True):
    print('Running FIRST: {}'.format(filepath_in))
    first_fpath_out = remove_ext(filepath_in) + '_first.nii.gz'

    # Create temporary path
    tmp_path = make_dirs(os.path.join(get_path(filepath_in), 'first_tmp'))
    t1_fpath = os.path.join(tmp_path, get_filename(filepath_in, ext=True))
    first_fpath_in = os.path.join(tmp_path, 't1_all_fast_firstseg.nii.gz')

    shutil.copy(filepath_in, t1_fpath)
    if is_skull_stripped:
        first_cmd_template = 'run_first_all -b -i {} -o t1'
    else:
        first_cmd_template = 'run_first_all -i {} -o t1'

    subprocess.check_call(['bash', '-c', first_cmd_template.format(t1_fpath)], cwd=tmp_path)

    first_nifti = nib.load(first_fpath_in)
    nib.Nifti1Image(
        (first_nifti.get_fdata() > 0.5).astype(float), first_nifti.affine, first_nifti.header
    ).to_filename(first_fpath_out)

    # Erase all temporary files
    shutil.rmtree(tmp_path)
    
def register_to_mni(t1_filepath, reference_filepath, transform_filepath_out, reg_filepath_out=None):
    """Registers the image to MNI space and stores the transform to MNI"""
    register_cmd = 'flirt -in {} -ref {} -omat {} '.format(t1_filepath, reference_filepath, transform_filepath_out)
    if reg_filepath_out is not None:
        register_cmd += '-out {} '.format(reg_filepath_out)
    register_opts = '-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear'
    subprocess.check_output(['bash', '-c', register_cmd + register_opts])


def segment_tissue(filepath_in):
    """Performs 3 tissue segmentation"""

    print('Running FAST: {}'.format(filepath_in))
    subprocess.check_call(['bash', '-c', 'fast {}'.format(filepath_in)])

    pve_fpaths = [remove_ext(filepath_in) + '_pve_{}.nii.gz'.format(i) for i in range(3)]
    out_fpaths = [remove_ext(filepath_in) + '_fast_{}.nii.gz'.format(i) for i in range(3)]

    for pve_fpath, out_fpath in zip(pve_fpaths, out_fpaths):
        os.rename(pve_fpath, out_fpath)

    # Remove all other files
    os.remove(os.path.join(remove_ext(filepath_in) + '_mixeltype.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_pveseg.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_seg.nii.gz'))


def run_fsl_bet(t1_fp, brain_out_fp=None, brain_mask_out_fp=None, skull_out_fp=None, betopts='-R -S'):
    with tempfile.TemporaryDirectory() as tempdir:
        t1_fp_in = os.path.join(tempdir, 't1.nii.gz')
        brain_fp = os.path.join(tempdir, 't1_bet_brain.nii.gz')
        brain_mask_fp = os.path.join(tempdir, 't1_bet_brain_mask.nii.gz')
        skull_fp = os.path.join(tempdir, 't1_bet_brain_skull.nii.gz')

        shutil.copy(t1_fp, t1_fp_in)
        subprocess.check_output(['bash', '-c', f'bet {t1_fp_in} t1_bet_brain -s {betopts}'], cwd=tempdir)

        if brain_out_fp is not None:
            shutil.copy(brain_fp, brain_out_fp)

        if brain_mask_out_fp is not None:
            shutil.copy(brain_mask_fp, brain_mask_out_fp)

        if skull_out_fp is not None:
            shutil.copy(skull_fp, skull_out_fp)


def run_fsl_pairreg(ref_brain_fp, mov_brain_fp, ref_skull_fp, mov_skull_fp, transform_fp):
    subprocess.check_output(
        ['bash', '-c', f'pairreg {ref_brain_fp} {mov_brain_fp} {ref_skull_fp} {mov_skull_fp} {transform_fp}'])

def apply_fsl_transform(filepath_in, filepath_ref, filepath_out, filepath_transform, fsldir='/usr/local/fsl'):
    tx_cmd = '{}/bin/flirt -out {} -applyxfm -init {} -ref {} -in {}'.format(
        fsldir, filepath_out, filepath_transform, filepath_ref, filepath_in)
    subprocess.check_output(['bash', '-c', tx_cmd])


def invert_fsl_transform(filepath_transform, filepath_inverse, fsldir='/usr/local/fsl'):
    subprocess.check_output(
        ['bash', '-c', f'{fsldir}/bin/convert_xfm -inverse -omat {filepath_inverse} {filepath_transform}'])


def run_fsl_pairreg_halfway_registration(
        baseline_fpath,
        followup_fpath,
        baseline_brain_fpath,
        followup_brain_fpath,
        baseline_skull_fpath=None,
        followup_skull_fpath=None,
        baseline_half_fpath_out=None,
        followup_half_fpath_out=None,
        baseline_to_half_fpath_out=None,
        followup_to_half_fpath_out=None,
        fsldir='/usr/local/fsl'
    ):

    with tempfile.TemporaryDirectory() as tmpdir:
        # First compute brains or skulls for both images if not given
        if baseline_brain_fpath is None or baseline_skull_fpath is None:
            if baseline_brain_fpath is None:
                baseline_brain_fpath = bet_baseline_brain_fpath = os.path.join(tmpdir, 'baseline_brain.nii.gz')
            else:
                bet_baseline_brain_fpath = None

            if baseline_skull_fpath is None:
                baseline_skull_fpath = bet_baseline_skull_fpath = os.path.join(tmpdir, 'baseline_skull.nii.gz')
            else:
                bet_baseline_skull_fpath = None

            run_fsl_bet(t1_fp=baseline_fpath,
                        brain_out_fp=bet_baseline_brain_fpath,
                        skull_out_fp=bet_baseline_skull_fpath,
                        betopts='-R -S -B')

        if followup_brain_fpath is None or followup_skull_fpath is None:
            if followup_brain_fpath is None:
                followup_brain_fpath = bet_followup_brain_fpath = os.path.join(tmpdir, 'followup_brain.nii.gz')
            else:
                bet_followup_brain_fpath = None

            if followup_skull_fpath is None:
                followup_skull_fpath = bet_followup_skull_fpath = os.path.join(tmpdir, 'followup_skull.nii.gz')
            else:
                bet_followup_skull_fpath = None

            run_fsl_bet(t1_fp=followup_fpath,
                        brain_out_fp=bet_followup_brain_fpath,
                        skull_out_fp=bet_followup_skull_fpath,
                        betopts='-R -S -B')

        # Now execute asymmetric pairreg registration forwards and backwards
        baseline_to_followup_fpath = os.path.join(tmpdir, 'baseline_to_followup.mat')
        run_fsl_pairreg(followup_brain_fpath, baseline_brain_fpath,
                        followup_skull_fpath, baseline_skull_fpath,
                        baseline_to_followup_fpath)

        followup_to_baseline_fpath = os.path.join(tmpdir, 'followup_to_baseline.mat')
        run_fsl_pairreg(baseline_brain_fpath, followup_brain_fpath,
                        baseline_skull_fpath, followup_skull_fpath,
                        followup_to_baseline_fpath)

        ### Now compute halfway transforms and transform images
        # Declare input and output filepaths
        B, F = baseline_fpath, followup_fpath
        Bbrain, Fbrain = baseline_brain_fpath, followup_brain_fpath
        B2F, F2B = baseline_to_followup_fpath, followup_to_baseline_fpath # F, B
        B2h, F2h = os.path.join(tmpdir, 'B2h.mat'), os.path.join(tmpdir, 'F2h.mat')
        # Declare temporary filepaths
        B2F_F2B = os.path.join(tmpdir, 'tmp_B2F_then_F2B.mat')
        B2F_F2B_scale = os.path.join(tmpdir, 'tmp_B2F_then_F2B.avscale')
        B2F_F2B_halfback = os.path.join(tmpdir, 'tmp_B2F_then_F2B_halfback.mat')
        F2B_scale = os.path.join(tmpdir, 'F2B.mat_avscale')
        # Run the commands to create the halfway transforms
        for cmd in [# replace both transforms with "average" (reduces error level AND makes system symmetric)
                    f'{fsldir}/bin/convert_xfm -concat {F2B} -omat {B2F_F2B} {B2F}',
                    f'{fsldir}/bin/avscale {B2F_F2B} {B} > {B2F_F2B_scale}',
                    f'{fsldir}/bin/extracttxt Backward {B2F_F2B_scale} 4 1 > {B2F_F2B_halfback}',
                    f'{fsldir}/bin/convert_xfm -concat {B2F_F2B_halfback} -omat {B2F} {B2F}',
                    f'{fsldir}/bin/convert_xfm -inverse -omat {F2B} {B2F}',
                    # replace the .mat matrix that takes 2->1 with 2->halfway and 1->halfway
                    f'{fsldir}/bin/avscale {F2B} {Bbrain} > {F2B_scale}',
                    f'{fsldir}/bin/extracttxt Forward {F2B_scale} 4 1 > {F2h}',
                    f'{fsldir}/bin/extracttxt Backward {F2B_scale} 4 1 > {B2h}']:
            subprocess.check_output(['bash', '-c', cmd])

        ### Copy results
        if baseline_to_half_fpath_out is not None:
            os.makedirs(os.path.dirname(baseline_to_half_fpath_out), exist_ok=True)
            shutil.copy(B2h, baseline_to_half_fpath_out)

        if followup_to_half_fpath_out is not None:
            os.makedirs(os.path.dirname(followup_to_half_fpath_out), exist_ok=True)
            shutil.copy(F2h, followup_to_half_fpath_out)

        if baseline_half_fpath_out is not None:
            tx_cmd = f'{fsldir}/bin/flirt -out {baseline_half_fpath_out} -applyxfm -init {B2h} -ref {B} -in {B}'
            subprocess.check_output(['bash', '-c', tx_cmd])

        if followup_half_fpath_out is not None:
            tx_cmd = f'{fsldir}/bin/flirt -out {followup_half_fpath_out} -applyxfm -init {F2h} -ref {B} -in {F}'
            subprocess.check_output(['bash', '-c', tx_cmd])
