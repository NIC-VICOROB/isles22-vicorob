import os
import json
import traceback
import numpy as np
import acglib as acg
import nibabel as nib
import pandas as pd

from tqdm import tqdm
from scipy.ndimage import label

from eval_utils_official import (compute_dice, compute_absolute_lesion_difference, 
                                 compute_absolute_volume_difference, compute_lesion_f1_score)


def print_isles22_metrics(measures_dict):
    for exp_name, exp_measures in measures_dict.items():
        pass    



def compute_isles22_measures(prob_pred_nifti_dict, msk_nifti_dict, threshold, min_vol, measures_fpath=None):
    if measures_fpath is not None and os.path.isfile(measures_fpath):
        with open(measures_fpath, 'r') as f:
            return json.load(f)

    isles22_measures = {}
    for case_id in tqdm(prob_pred_nifti_dict.keys(), desc='Computing measures'):
        pred_nifti = prob_pred_nifti_dict[case_id]
        msk_nifti = msk_nifti_dict[case_id]
        
        msk_arr = msk_nifti.get_fdata().astype(int)

        # 1. Threshold and min lesion size
        pred_arr_bin = (pred_nifti.get_fdata() >= threshold).astype(int)

        pred_nifti.uncache()
        msk_nifti.uncache()

        if min_vol == 0.0:
            pred_arr = pred_arr_bin
        else:
            pred_labels, nl = label(pred_arr_bin, structure=np.ones((3,3,3)))
            pred_arr = np.zeros_like(pred_arr_bin)
            for i in range(1, nl + 1):
                les_vol = np.prod(pred_nifti.header['pixdim'][1:4]) * np.count_nonzero(pred_labels == i) 
                if les_vol >= min_vol:
                    pred_arr[pred_labels == i] = 1
            
        ### MEASURES
        try:
            isles22_measures[case_id] = compute_binary_seg_measures(
                pred=pred_arr,
                gt=msk_arr,
                nifti_gt=msk_nifti)
        except ZeroDivisionError as e:
            print(case_id)
            traceback.print_exc()

    if measures_fpath is not None:
        with open(measures_fpath, 'w') as f:
            json.dump(isles22_measures, f)

    return isles22_measures





def compute_binary_seg_measures(pred, gt, nifti_gt):
    pred = (pred > 0.5).astype(int)
    gt = (gt > 0.5).astype(int)

    # GENERAL
    tp = np.count_nonzero(np.logical_and(gt == 1, pred == 1))
    #tn = np.count_nonzero(np.logical_and(gt == 0, pred == 0))
    fn = np.count_nonzero(np.logical_and(gt == 1, pred == 0))
    fp = np.count_nonzero(np.logical_and(gt == 0, pred == 1))
    
    if tp + fn > 0:
        sensitivity = tp / (tp + fn)    
    else:
        sensitivity = 0.0

    if tp + fp > 0:
        ppv = tp / (tp + fp)
    else:
        ppv = 1.0

    # Using the official evaluation functions
    f1_lesion, tp_les, fp_les, fn_les = compute_lesion_f1_score(gt, pred)

    lesion_sensitivity = tp_les / (tp_les + fn_les) if (tp_les + fn_les) > 0 else 0.0
    lesion_ppv = tp_les / (tp_les + fp_les) if (tp_les + fp_les) > 0 else 1.0

    # Compute performance metrics.
    voxel_volume = np.prod(nifti_gt.header.get_zooms())/1000 # Get voxel volume

    return {
        'I22_dsc': compute_dice(gt, pred),
        'I22_abs_vol_diff': compute_absolute_volume_difference(gt, pred, voxel_volume),
        'I22_abs_lesion_count_diff': compute_absolute_lesion_difference(gt, pred),
        'I22_f1_lesion': f1_lesion,
        'sensitivity': float(sensitivity),
        'ppv': ppv,
        'lesion_sensitivity': lesion_sensitivity,
        'lesion_ppv': lesion_ppv
    }


if __name__ == '__main__':
    compute_isles22_measures(
        prob_pred_dict={'test1': '/home/albert/Desktop/ISLES22/datasets/prepared/sub-strokecase0001_ses-0001/pred_test.nii.gz'},
        msk_dict={'test1': '/home/albert/Desktop/ISLES22/datasets/prepared/sub-strokecase0001_ses-0001/gt_test.nii.gz'},
        threshold=0.1,
        min_vol=0.1
    )