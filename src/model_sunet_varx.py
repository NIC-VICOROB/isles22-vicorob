import sys
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data

INIT_PRELU = 0.0
BN_MOMENTUM = 0.01


class SUNETx4_varX(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        x_dim_downsamplings=(2, 2, 2),
        nfilts=32,
        dropout_rate=0.0,
        activation=None
    ):
        super(SUNETx4_varX, self).__init__()

        self.inconv = nn.Conv3d(in_ch, 1 * nfilts, 3, padding=1)
        
        self.dual1 = DualRes3D(1 * nfilts)
        self.dual2 = DualRes3D(2 * nfilts, dropout_rate=dropout_rate)
        self.dual3 = DualRes3D(4 * nfilts, dropout_rate=dropout_rate)
        self.dual4 = DualRes3D(8 * nfilts, dropout_rate=dropout_rate)
        
        self.down1 = VariableDownStep3D(1 * nfilts, downsample_factors=(x_dim_downsamplings[0], 2, 2))
        self.down2 = VariableDownStep3D(2 * nfilts, downsample_factors=(x_dim_downsamplings[1], 2, 2))
        self.down3 = VariableDownStep3D(4 * nfilts, downsample_factors=(x_dim_downsamplings[2], 2, 2))
        
        self.mono3 = MonoRes3D(4 * nfilts)
        self.mono2 = MonoRes3D(2 * nfilts)
        self.mono1 = MonoRes3D(1 * nfilts)
        
        self.up4 = VariableUpStep3D(in_ch=8 * nfilts, out_ch=4 * nfilts, upsample_factors=(x_dim_downsamplings[2], 2, 2))
        self.up3 = VariableUpStep3D(in_ch=4 * nfilts, out_ch=2 * nfilts, upsample_factors=(x_dim_downsamplings[1], 2, 2))
        self.up2 = VariableUpStep3D(in_ch=2 * nfilts, out_ch=1 * nfilts, upsample_factors=(x_dim_downsamplings[0], 2, 2))

        self.outconv = nn.Conv3d(nfilts, out_ch, 3, padding=1)
        self.activation_out = activation
        
    def forward(self, x_in):
        l1_start = self.inconv(x_in)
        
        l1_end = self.dual1(l1_start)
        l2_start = self.down1(l1_end)
        
        l2_end = self.dual2(l2_start)
        l3_start = self.down2(l2_end)
        
        l3_end = self.dual3(l3_start)
        l4_start = self.down3(l3_end)
        
        l4_latent = self.dual4(l4_start)
        r4_up = self.up4(l4_latent)
        
        r3_start = l3_end + r4_up
        r3_end = self.mono3(r3_start)
        r3_up = self.up3(r3_end)
        
        r2_start = l2_end + r3_up
        r2_end = self.mono2(r2_start)
        r2_up = self.up2(r2_end)
        
        r1_start = l1_end + r2_up
        r1_end = self.mono1(r1_start)
        
        pred = self.outconv(r1_end)

        # print('l1_start.size()', l1_start.size())
        # print('l2_start.size()', l2_start.size())
        # print('l3_start.size()', l3_start.size())
        # print('l4_latent.size()', l4_latent.size())
        # print('r4_up.size()', r4_up.size())
        # print('l3_end.size()', r4_up.size())
        # print('r3_up.size()', r3_up.size())
        # print('r2_up.size()', r2_up.size())
        # print('pred.size()', pred.size())

        if self.activation_out is not None:
            pred = self.activation_out(pred)
        
        return pred

class DualRes3D(nn.Module):
    def __init__(self, num_ch, dropout_rate=0.0):
        super(DualRes3D, self).__init__()
        Dropout = nn.AlphaDropout
        
        self.conv_path = nn.Sequential(
                nn.BatchNorm3d(num_ch, momentum=BN_MOMENTUM, eps=0.001),
                nn.PReLU(num_ch, init=INIT_PRELU),
                nn.Conv3d(num_ch, num_ch, 3, padding=1),
                Dropout(p=dropout_rate),
                nn.BatchNorm3d(num_ch, momentum=BN_MOMENTUM, eps=0.001),
                nn.PReLU(num_ch, init=INIT_PRELU),
                nn.Conv3d(num_ch, num_ch, 3, padding=1))
    
    def forward(self, x_in):
        return self.conv_path(x_in) + x_in


class MonoRes3D(nn.Module):
    def __init__(self, num_ch):
        super(MonoRes3D, self).__init__()
        self.conv_path = nn.Sequential(
                nn.BatchNorm3d(num_ch, momentum=BN_MOMENTUM, eps=0.001),
                nn.PReLU(num_ch, init=INIT_PRELU),
                nn.Conv3d(num_ch, num_ch, 3, padding=1))
    
    def forward(self, x_in):
        x_out = self.conv_path(x_in) + x_in
        return x_out


class VariableDownStep3D(nn.Module):
    def __init__(self, in_ch, downsample_factors):
        super(VariableDownStep3D, self).__init__()

        self.pool_path = nn.MaxPool3d(kernel_size=downsample_factors)
        self.conv_path = nn.Sequential(
                nn.BatchNorm3d(in_ch, momentum=BN_MOMENTUM, eps=0.001),
                nn.PReLU(in_ch, init=INIT_PRELU),
                nn.Conv3d(
                    in_channels=in_ch, 
                    out_channels=in_ch, 
                    kernel_size=[{2: 3, 3: 5}[df] for df in downsample_factors],
                    padding=[{2: 1, 3: 2}[df] for df in downsample_factors],
                    stride=[{2: 2, 3: 3}[df] for df in downsample_factors]))
    
    def forward(self, x_in):
        x_out = torch.cat((self.conv_path(x_in), self.pool_path(x_in)), dim=1)  # Channel dimension
        return x_out


class VariableUpStep3D(nn.Module):
    def __init__(self, in_ch, out_ch, upsample_factors):
        super(VariableUpStep3D, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=[{2: 3, 3: 5}[uf] for uf in upsample_factors],
            padding=[{2: 1, 3: 2}[uf] for uf in upsample_factors],
            output_padding=[{2: 1, 3: 2}[uf] for uf in upsample_factors],
            stride=[{2: 2, 3: 3}[uf] for uf in upsample_factors])

    def forward(self, x_in):
        return self.conv_transpose(x_in)

if __name__ == '__main__':
    mdl = SUNETx4_varX(3, 1, x_dim_downsamplings=(3,3,2))
    mdl(torch.ones((1, 3, 162, 24, 24)))


    


    
