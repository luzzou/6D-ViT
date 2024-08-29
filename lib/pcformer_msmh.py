from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# helpers



def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

LayerNorm = partial(nn.InstanceNorm1d, affine = True)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv1d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv1d(dim, dim * 2, 1, bias = False)
        self.to_out = nn.Conv1d(dim, dim, 1, bias = False)

    def forward(self, x):
        # h, w = x.shape[-2:]
        # heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))  
        # q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  #bs,32,32
        attn = sim.softmax(dim = -1) 
    

        out = einsum('b i j, b j d -> b i d', attn, v)  #bs,32,256
        # out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 1),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):

            overlap_patch_embed = nn.Conv1d(dim_in, dim_out, 1)
            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                overlap_patch_embed,
                layers
            ]))

    def forward(self, x, return_layer_outputs = False):

        layer_outputs = []
        for (overlap_embed, layers) in self.stages:
            #origin: bs,64,1024
            x = overlap_embed(x) #bs,32,1024
            for (attn, ff) in layers:
                x = attn(x) + x  #bs,32,1024
                x = ff(x) + x  #bs,32,1024

            # print(x.shape)
            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret


class PCformer(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 64,
        decoder_dim = 256,
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, decoder_dim, 1),
         
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv1d(4 * decoder_dim, decoder_dim, 1),
        )

    def forward(self, x):  #bs,3,256,256  image_dim

        layer_outputs = self.mit(x, return_layer_outputs = True)  
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)] 
        fused = torch.cat(fused, dim = 1)  
        out = self.to_segmentation(fused) 

        return out



