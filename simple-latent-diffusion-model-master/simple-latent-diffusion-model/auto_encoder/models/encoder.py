#source : https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L368
import torch
import torch.nn as nn

from auto_encoder.components.normalize import Normalize
from auto_encoder.components.resnet_block import ResnetBlock
from auto_encoder.components.sampling import Downsample
from auto_encoder.components.nonlinearity import nonlinearity

class Encoder(nn.Module):
    def __init__(self, *, in_channels, resolution, channels, channel_multipliers = (1, 2, 4, 8), z_channels, num_res_blocks,
                 dropout = 0.0, resample_with_conv : bool = True, double_z : bool = True):
        super().__init__()
        self.ch = channels
        self.num_resolutions = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.z_channels = z_channels
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size = 3, stride = 1, padding = 1)
        curr_res = resolution
        in_ch_mult = (1, ) + tuple(channel_multipliers)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * channel_multipliers[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels = block_in, out_channels = block_out, dropout = dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resample_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels = block_in, out_channels = block_in, dropout = dropout)
        self.mid.block_2 = ResnetBlock(in_channels = block_in, out_channels = block_in, dropout = dropout)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels,
                                        kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                
        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
        