#source : https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L368
import torch
import torch.nn as nn
import numpy as np
from auto_encoder.components.normalize import Normalize
from auto_encoder.components.resnet_block import ResnetBlock
from auto_encoder.components.sampling import Upsample
from auto_encoder.components.nonlinearity import nonlinearity

class Decoder(nn.Module):
    def __init__(self, *, in_channels, out_channels, resolution, channels, channel_multipliers = (1, 2, 4, 8), z_channels, num_res_blocks,
                 dropout = 0.0, resample_with_conv : bool = True):
        super().__init__()
        self.ch = channels
        self.num_resolutions = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.z_channels = z_channels
        
        in_ch_mult = (1 , ) + tuple(channel_multipliers)
        block_in = self.ch * in_ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1 , z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size = 3, stride = 1, padding = 1)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels = block_in, out_channels = block_in, dropout = dropout)
        self.mid.block_2 = ResnetBlock(in_channels = block_in, out_channels = block_in, dropout = dropout)
        
        # upsampling
        
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = self.ch * channel_multipliers[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels = block_in, out_channels = block_out,
                                         dropout = dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resample_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels,
                                        kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, z):
        assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        
        # z to block_in
        h = self.conv_in(z)
                
        # middle
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h