import os
import copy
import itertools

from tqdm import tqdm
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

import acglib as acg
from datamodule_isles22 import ISLES22_DataModule



class ISLES22_UnetModule(pl.LightningModule):
    def __init__(
            self,
            unet_model,
            activation,
            segmentation_loss,
            optimizer_class,
            optimizer_lr,
            optimizer_opts
    ):
        super().__init__()
        # Save __init__ arguments as hyperparameters to be stored later in checkpoints
        # self.save_hyperparameters() # JUST TAKES TOO MUCH TIME MAN

        self.model = unet_model
        self.activation = activation
        self.loss_fn = segmentation_loss
        
        self.optimizer_class = optimizer_class
        self.optimizer_opts = optimizer_opts
        self.lr = optimizer_lr

        
    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), lr=self.lr, **self.optimizer_opts)
    
    def forward(self, x):
        output = self.model(x)
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def training_step(self, batch, batch_idx):
        x, target = batch
        output = self.model(x)
        loss_batch = self.loss_fn(output, target)
        self.log('train_loss', loss_batch, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss_batch}
    
    def validation_step(self, batch, batch_idx):
        x, target = batch
        output = self.model(x)
        loss_batch = self.loss_fn(output, target)
        self.log('val_loss', loss_batch, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss_batch}
    
    def image_inference(
        self,
        image,
        nifti,
        datamodule : ISLES22_DataModule,
        device,
        probs_fpath=None,
        decimals=4
    ):
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
        self.eval()
        self.to(device)
        with torch.no_grad():
            for x, x_slices in tqdm(inference_dataloader, desc=os.path.basename(probs_fpath), leave=False):
                x = x.to(device)
                output = self(x)
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
                np.round(infered_probs[1], decimals=decimals), nifti.affine, nifti.header
            ).to_filename(probs_fpath)

        return infered_probs

        