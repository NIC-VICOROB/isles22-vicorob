import copy
import itertools
import os
from monai import transforms
import nibabel as nib
import pandas as pd
import acglib as acg
import numpy as np
import json
from tqdm import tqdm
import torch
import SimpleITK as sitk

from scipy.ndimage import label

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from utils_prepare import load_prepared_isles22_data, prepare_isles22_resample_case
from utils_inference import compute_normalization_parameters, normalize_array

def list_dirs(p):
    return list(sorted([f for f in os.scandir(p) if f.is_dir()], key=lambda f: f.name))

def list_files(p):
    return list(sorted([f for f in os.scandir(p) if f.is_file()], key=lambda f: f.name))

def load_isles22_original_dict(dataset_path):
    isles22_original_dict = {}

    data_path = os.path.join(dataset_path, 'rawdata')
    gt_path = os.path.join(dataset_path, 'derivatives')

    for subject_entry in list_dirs(data_path):
        session_entries = list_dirs(subject_entry.path)

        if len(session_entries) > 1:
            print('WARNING SUBJECT WITH MORE THAN ONE SESSION?!?!?!')

        for session_entry in session_entries:
            case_id = f'{subject_entry.name}_{session_entry.name}'
            isles22_original_dict[case_id] = {}
            # Load IMAGE modalities
            for modality_name in ['adc', 'dwi', 'flair']:
                image_filepath = os.path.join(session_entry.path, f'{case_id}_{modality_name}.nii.gz')
                meta_filepath = os.path.join(session_entry.path, f'{case_id}_{modality_name}.json')
                nifti = nib.load(image_filepath)
                try:
                    with open(meta_filepath, 'r') as fp:
                        meta = json.load(fp)
                except FileNotFoundError:
                    meta = None
                isles22_original_dict[case_id][modality_name] = {
                    'fpath': image_filepath, 'nifti': nifti, 'meta': meta}

            # Now load mask
            msk_filepath = os.path.join(gt_path, subject_entry.name, session_entry.name, f'{case_id}_msk.nii.gz')
            msk_nifti = nib.load(msk_filepath)
            isles22_original_dict[case_id]['msk'] = {
                    'fpath': msk_filepath,
                    'nifti': msk_nifti}

    return isles22_original_dict


def compute_crossval_splits(all_ids, num_folds, val_fraction):
    """
    Returns the crossvalidation splits as:

    crossval_splits = [
        { 'train': [1,2,3], 'val': [4,5,6], 'test': [7,8,9] }, # Fold 1
        { 'train': [1,2,3], 'val': [4,5,6], 'test': [7,8,9] }, # Fold 2

    ]
    """

    # Compute the test splits for crossvalidation
    num_test_per_fold = int(np.ceil(len(all_ids) / num_folds))
    ids_test_per_fold = [all_ids[i:i+num_test_per_fold] for i in range(0, len(all_ids), num_test_per_fold)]

    # Compute the test/fit split for this fold
    crossval_splits = []
    for fold_idx in range(num_folds):
        # First get the test ids
        test_ids = ids_test_per_fold[fold_idx]
        # Build the train and validation datasets from the remaining ids
        fit_ids = [fit_id for fit_id in all_ids if fit_id not in test_ids]
        train_ids, val_ids = acg.lists.split_list(fit_ids, fraction=1.0 - val_fraction)

        crossval_splits.append({'train': train_ids, 'val': val_ids, 'test': test_ids})

    return crossval_splits




class ISLES22_DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        isles22_original_dict,
        prepared_path, 
        template_fpath,
        num_folds,
        input_modalities,
        do_symmetric_modalities,
        patch_type, 
        patch_shape,
        num_patches, 
        sampling_fractions,
        norm_type,
        batch_size, 
        extraction_step,
        num_threads,
        augmentation_prob=0.0,
        val_fraction=0.2,
        crossval_splits=None,
        max_cases=None,
        min_patches_per_lesion=5
    ):  
        assert patch_type in ('strip', 'isotropic')

        super().__init__()
        self.original_dict = isles22_original_dict
        self.prepared_path = prepared_path
        self.template_fpath = template_fpath
        
        self.input_modalities = input_modalities
        self.do_symmetric_modalities = do_symmetric_modalities
        if self.do_symmetric_modalities:
            self.input_modalities = self.input_modalities + tuple([f'sym_{im}' for im in self.input_modalities])

        self.patch_type = patch_type
        self.patch_shape = patch_shape
        self.num_patches = num_patches
        self.sampling_fractions = sampling_fractions
        self.norm_type = norm_type
        self.batch_size = batch_size
        self.extraction_step = extraction_step
        self.num_threads = num_threads
        self.val_fraction = val_fraction

        self.max_cases = max_cases
        self.augment_prob = augmentation_prob
        self.min_patches_per_lesion = min_patches_per_lesion
        
        self.crossval_splits = crossval_splits
        self.num_folds = num_folds
        if crossval_splits is not None:
            self.num_folds = len(self.crossval_splits)

            print('Crossval splits')
            for n, cs in enumerate(self.crossval_splits):
                print(f'Fold {n}')
                print(f'    train: {len(cs[n]["train"])} [{cs[n]["train"][0]} -> {cs[n]["train"][-1]}]')
                print(f'      val: {len(cs[n]["val"])}   [{cs[n]["val"][0]} -> {cs[n]["val"][-1]}]')
                print(f'     test: {len(cs[n]["test"])}  [{cs[n]["test"][0]} -> {cs[n]["test"][-1]}]')


        self.fold_idx = 0
        self.prepared_data = None

    def prepare_data(self):
        """Structure and preprocess files"""
        super().prepare_data()

        acg.parallel_run(
            func=prepare_isles22_resample_case,
            args=[[case_dict, os.path.join(self.prepared_path, case_id), self.template_fpath] 
                for case_id, case_dict in self.original_dict.items()],
            num_threads=self.num_threads,
            do_print_progress_bar=True,
            progress_bar_prefix='Running prepare_data()... ',
            progress_bar_suffix='cases prepared')
        

    def setup(self, stage=None):
        """Load data (if not loaded)"""
        print('Running setup()...')

        # Load prepared_data if not already loaded
        if self.prepared_data is None:
            self.prepared_data = load_prepared_isles22_data(
                self.num_threads, 
                self.prepared_path, 
                self.input_modalities,
                max_cases=self.max_cases)

            # CC lesion volumes
            from scipy.ndimage import label
            lesion_volumes = []
            for case_id, case_dict in self.prepared_data.items():
                lbls, nles = label((case_dict['msk'][0] > 0.5).astype(int), structure=np.ones((3,3,3)))

                for i in range(1, nles + 1):
                    les_vol = np.prod(case_dict['nifti_ref'].header['pixdim'][1:4]) * np.count_nonzero(lbls == i) 
                    lesion_volumes.append(les_vol)

            print('PREPARED LESION VOLUME DISTRIBUTION')
            print(pd.Series(lesion_volumes).describe())

        # Compute the train/val/test splits for crossvalidation
        if self.crossval_splits is None:
            self.crossval_splits = compute_crossval_splits(
                all_ids=list(self.prepared_data.keys()),
                num_folds=self.num_folds,
                val_fraction=self.val_fraction)
        
            print('Crossval splits')
            for n, cs in enumerate(self.crossval_splits):
                print(f'Fold {n}')
                print(f'    train: {len(cs["train"])} [{cs["train"][0]} -> {cs["train"][-1]}]')
                print(f'      val: {len(cs["val"])}   [{cs["val"][0]} -> {cs["val"][-1]}]')
                print(f'     test: {len(cs["test"])}  [{cs["test"][0]} -> {cs["test"][-1]}]')

    
    def set_fold_idx(self, fold_idx):
        assert 0 <= fold_idx < self.num_folds
        self.fold_idx = fold_idx

    def train_dataloader(self):
        train_ids = self.crossval_splits[self.fold_idx]['train']
        train_data = {k: v for k, v in self.prepared_data.items() if k in train_ids}

        # Extract the fit instructions for the train set
        num_train_patches = int(np.floor(self.num_patches * (1.0 - self.val_fraction)))
        num_patches_per_case = int(np.floor(num_train_patches / len(train_data))) # Num patches per image

        train_instructions = acg.parallel_run(
            func=self.generate_case_fit_instructions,
            args=[[case_id, case_dict, num_patches_per_case] for case_id, case_dict in train_data.items()],
            num_threads=self.num_threads,
            do_print_progress_bar=True,
            progress_bar_prefix='Generating train instructions')
        train_instructions = list(itertools.chain.from_iterable(train_instructions))
        
        train_dataset = acg.generators.InstructionDataset(
                    instructions=train_instructions,
                    data=train_data,
                    get_item_func=self.extract_fit_patch)

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_threads)

    def val_dataloader(self):
        val_ids = self.crossval_splits[self.fold_idx]['val']
        val_data = {k: v for k, v in self.prepared_data.items() if k in val_ids}
        
        # Extract the fit instructions for the val set
        num_val_patches = int(np.floor(self.num_patches * self.val_fraction))
        num_patches_per_case = int(np.floor(num_val_patches / len(val_data))) # Num patches per image
        val_instructions = acg.parallel_run(
            func=self.generate_case_fit_instructions,
            args=[[case_id, case_dict, num_patches_per_case] for case_id, case_dict in val_data.items()],
            num_threads=self.num_threads,
            do_print_progress_bar=True,
            progress_bar_prefix='Generating val instructions')
        val_instructions = list(itertools.chain.from_iterable(val_instructions))
        
        val_dataset = acg.generators.InstructionDataset(
                    instructions=val_instructions,
                    data=val_data,
                    get_item_func=self.extract_fit_patch)

        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_threads)

    def get_test_data(self):
        test_ids = self.crossval_splits[self.fold_idx]['test']
        test_data = {k: v for k, v in self.prepared_data.items() if k in test_ids}
        return test_data

    def generate_case_fit_instructions(
        self,
        case_id, 
        case_dict, 
        num_patches_case
    ):
        if self.patch_type == 'strip':
            raise NotImplementedError

        norm_params = compute_normalization_parameters(
            case_dict['image'], norm_type=self.norm_type, ignore_background_label=0)

        msk = (case_dict['msk'][0] > 0.5).astype(int)
        num_patches_lesion = np.ceil(self.sampling_fractions[2] * num_patches_case)

        ### PHASE 1 - LESION SAMPLING
        lesions_labelled, num_lesions = label(msk, structure=np.ones((3,3,3)))
        num_patches_min_lesion = min(self.min_patches_per_lesion * num_lesions, num_patches_lesion)

        if num_lesions > 0:
            centers_min_lesion = acg.patch.sample_centers_balanced(
                labels_image=lesions_labelled,
                n=num_patches_min_lesion,
                patch_shape=self.patch_shape,
                add_rand_offset=False,
                exclude=[0])
        else:
            centers_min_lesion = []

        ### PHASE 2 - GENERAL SAMPLING
        # Parenchyma and lesion labels and sample fractioned 1% 49% 50%
        sampling_labels = (np.max(case_dict['image'], axis=0) > 0.0).astype(int)
        sampling_labels[msk == 1] = 2
        
        centers_general = acg.patch.sample_centers_labels_number(
            labels_image=sampling_labels,
            labels_num_centers={0: int(np.ceil(self.sampling_fractions[0] * num_patches_case)),
                                1: int(np.ceil(self.sampling_fractions[1] * num_patches_case)),
                                2: max(num_patches_lesion - num_patches_min_lesion, 0)},
            patch_shape=self.patch_shape,
            add_rand_offset=True)

        centers = centers_min_lesion + centers_general

        case_instructions = \
            [{'case_id': case_id,
              'center': center,
              'do_symmetric_modality_augmentation': self.do_symmetric_modalities,
              'patch_shape': self.patch_shape,
              'augment_prob': self.augment_prob,
              'norm_type': self.norm_type,
              'norm_params': norm_params} for center in centers]

        return case_instructions


    @staticmethod
    def extract_fit_patch(instr, data):
        case_dict = data[instr['case_id']]

        # Extract patch
        image_patch = acg.patch.get_patch(case_dict['image'], instr['center'], instr['patch_shape'])
        msk_patch = acg.patch.get_patch(case_dict['msk'], instr['center'], instr['patch_shape'])

        # Normalize image_patch
        image_patch = normalize_array(image_patch, norm_params=instr['norm_params'], norm_type=instr['norm_type'])

        # TODO more data augmentation? Noise! Intensity shifts!
        augment_func = transforms.Compose([
            transforms.RandFlip(prob=instr['augment_prob'], spatial_axis=1)
        ])
        image_patch = augment_func(image_patch) # AUGMENT AFTER NORMALIZATION!!!

        # Convert to torch
        image_patch_torch = torch.tensor(np.ascontiguousarray(image_patch), dtype=torch.float32)
        gt_patch_torch = torch.tensor(np.ascontiguousarray(msk_patch), dtype=torch.float32)
        
        return (image_patch_torch, gt_patch_torch)



    def image_inference_dataloader(self, image):
        """Custom function to perform patch-based inference over the whole image"""

        norm_params = compute_normalization_parameters(image, norm_type=self.norm_type, ignore_background_label=0)
        
        # TODO adjust for patch_type

        if self.patch_type == 'isotropic':
            patch_centers = acg.patch.sample_centers_uniform(
                image_shape=image.shape[1:],
                step=self.extraction_step, 
                patch_shape=self.patch_shape,
                ensure_full_coverage=True)
        elif self.patch_type == 'strip':
            pass
        else:
            raise ValueError(f'Patch type {self.patch_type} not recognized')

        instructions = [{
                'center': center,
                'patch_shape': self.patch_shape,
                'do_symmetric_modality_augmentation': self.do_symmetric_modalities,
                'norm_type': self.norm_type,
                'norm_params': norm_params
            } for center in patch_centers]

        inference_dataset = acg.generators.InstructionDataset(
            instructions=instructions,
            data=image,
            get_item_func=self.extract_inference_patch)

        return DataLoader(dataset=inference_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=lambda batch: (torch.stack([b[0] for b in batch], dim=0), [b[1] for b in batch]))
    
    @staticmethod
    def extract_inference_patch(instr, image):
        # Get patch_slices
        patch_slices = acg.patch.get_patch_slices(image.ndim, instr['center'], instr['patch_shape'])
        # Extract patch from image
        in_patch = copy.deepcopy(image[patch_slices])
        # Normalize image_patch
        in_patch = normalize_array(in_patch, instr['norm_params'], instr['norm_type'])

        # Concatenate image and lesion patches
        input_torch = torch.tensor(np.ascontiguousarray(in_patch), dtype=torch.float32)
        return input_torch, patch_slices
