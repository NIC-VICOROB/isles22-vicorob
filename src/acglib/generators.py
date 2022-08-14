import copy

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader


def construct_dataloader(*args, **kwargs):
    return DataLoader(*args, **kwargs)


class InstructionDataset(TorchDataset):
    def __init__(self, instructions, data, get_item_func):
        assert callable(get_item_func)
        self.instr = instructions
        self.data = data
        self.get_item = get_item_func
        
    def __len__(self):
        return len(self.instr)

    def __getitem__(self, idx):
        return self.get_item(self.instr[idx], self.data)


def get_normalize_params(arr, ntype, mask=None):
    norm_img = copy.deepcopy(arr)
    if mask is not None:
        norm_img[mask == 0] = np.nan

    if ntype == '0mean1std':
        means = np.nanmean(norm_img, axis=(-3, -2, -1), keepdims=True, dtype=np.float64)
        stds = np.nanstd(norm_img, axis=(-3, -2, -1), keepdims=True, dtype=np.float64)
    elif ntype == 'minmax':
        pmin, pmax = np.nanpercentile(norm_img, [0.1, 99.9], axis=(-3, -2, -1), keepdims=True)
        means = (pmax - pmin) / 2.0
        stds = (pmax - pmin) / 2.0 # Aprox to the -2, 2 range
    else:
        raise ValueError

    del norm_img
    return means, stds


def normalize(arr, m, s):
    return (arr - m) / s


if __name__ == '__main__':
    import nibabel as nib
    import matplotlib.pyplot as plt

    a = nib.load('/home/albert/Desktop/docker_tests/xnat_vicorob_E00365/t1-preprocessed.nii.gz').get_fdata()

    m1, s1 = get_normalize_params(a, '0mean1std', mask=(a > 0))
    m2, s2 = get_normalize_params(a, 'minmax', mask=(a > 0))

    a1 = normalize(a, m1, s1)
    a2 = normalize(a, m2, s2)

    bins = np.arange(-3.0, 3.0, 0.05)
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(a1[np.nonzero(a)], bins=bins)
    axs[1].hist(a2[np.nonzero(a)], bins=bins)
    plt.show()
