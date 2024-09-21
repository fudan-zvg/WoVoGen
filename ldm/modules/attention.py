from inspect import isfunction
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
from torch.cuda.amp import autocast

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        if mask is not None:
            mask = (1 - mask.reshape(b, -1, q.shape[1]))  # B,2,M
            mask = mask.unsqueeze(-1).repeat(1,1,1,self.heads).permute(0,3,1,2).reshape(b*self.heads, -1, q.shape[1]).contiguous()
            k_len = k.shape[1]
            mask_len = k_len + (8-(k_len%8))
            mask_ = torch.zeros(b*self.heads, q.shape[1], mask_len, device=q.device, dtype=q.dtype)

            for i in range(k_len//77):
                mask_[:,:,i*77:(i+1)*77] = mask[:,i,...].unsqueeze(-1).repeat(1, 1, 77)
            mask_[mask_!=0] = -math.inf
            mask = mask_[:,:,:k_len+1]

            k = torch.cat([k, torch.zeros(b*self.heads, 1, k.shape[2], device=q.device, dtype=q.dtype)], dim=1)
            v = torch.cat([v, torch.zeros(b*self.heads, 1, v.shape[2], device=q.device, dtype=q.dtype)], dim=1)

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask, op=self.attention_op)

        # if exists(mask):
        #     # raise NotImplementedError
        #     out = torch.nan_to_num(out, nan=0.0)
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

class MultiViewCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        # context_dim = default(context_dim, query_dim)
        context_dim = query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(zero_module(nn.Linear(inner_dim, query_dim)), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, xl, xr):
        q = self.to_q(x)
        context = torch.cat([xl, xr], dim=1)
        b, _, _ = q.shape
        
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, use_multi_view_attn=False, use_local=False, with_position=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.norm4 = nn.LayerNorm(dim)
        self.use_multi_view_attn = use_multi_view_attn
        if self.use_multi_view_attn:
            self.multi_view_attn = MultiViewCrossAttention(query_dim=dim, context_dim=context_dim,
                                            heads=n_heads, dim_head=d_head, dropout=dropout)
        self.use_local = use_local

        # for position embedding
        self.with_position = with_position
        if self.with_position:
            self.depth_num = 48
            self.position_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
            self.depth_start = 0.
            self.position_dim = 3 * self.depth_num
            self.embed_dims = dim
            
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x, context=None, meta=None):
        if meta is not None:
            self.is_train = meta['is_train']
            self.hw = meta['hw'][x.shape[1]]
            if self.with_position:
                self.img2ego = meta['img2ego']
            if self.use_local:
                self.msk = meta['msk']
                self.msk_txt = meta['msk_txt']
        return checkpoint(self._forward, (x, context,), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        if self.use_local:
            msk_ = self.msk[self.hw[-1]]
            msk_txt_ = [self.msk_txt[:,i,...] for i in range(self.msk_txt.shape[1])]

            msk_txt_ = torch.cat(msk_txt_, dim=1)
            x_ = self.attn2(self.norm2(x), context=msk_txt_, mask=msk_)  # .reshape(B,D,C)
            # x_ = torch.nan_to_num(x_, nan=0.0)
            x = x + x_ 
        if self.use_multi_view_attn:
            if self.is_train:
                b, d, c = x.shape
                x_emb = x.clone()
                if self.with_position:
                    h, w = self.hw
                    img_feats_shape = (b//3, 3, c, h, w)
                    img_size = (256, 448)
                    pos_emb = self.position_embeding(img_feats_shape, self.img2ego, img_size, device=x.device)  # b//3, 3, c, h, w
                    x_emb = x + pos_emb.reshape(b, c, d).permute(0, 2, 1)
                    
                x_emb_ = self.norm4(x_emb)
                x_emb_ = x_emb_.reshape(b//3, 3, d, c)

                xcur,xl,xr = x_emb_[:,0].clone() ,x_emb_[:,1].clone(),x_emb_[:,2].clone()

                # xl = rearrange(xl, 'b (h w) c -> b h w c', h=self.hw[0], w=self.hw[1])
                # xl = xl[:, :, 3*self.hw[1]//4:, :]
                # xl = rearrange(xl, 'b h w c -> b (h w) c')
                # xr = rearrange(xr, 'b (h w) c -> b h w c', h=self.hw[0], w=self.hw[1])
                # xr = xr[:, :, :self.hw[1]//4, :]
                # xr = rearrange(xr, 'b h w c -> b (h w) c')

                xcur = self.multi_view_attn(xcur, xl, xr)
                x = x.reshape(b//3, 3, d, c)
                x[:, 0] = xcur + x[:, 0]
                x = x.reshape(b, d, c)
            else:
                b, d, c = x.shape
                # x = x.reshape(b//6, 6, d, c)
                x_emb = x.clone()
                if self.with_position:
                    h, w = self.hw
                    img_feats_shape = (b//6, 6, c, h, w)
                    img_size = (256, 448)
                    pos_emb = self.position_embeding(img_feats_shape, self.img2ego, img_size, device=x.device)  # b//6, 6, c, h, w
                    x_emb = x + pos_emb.reshape(b, c, d).permute(0, 2, 1)

                x_emb = self.norm4(x_emb)
                # x_emb = x_emb.reshape(b//6, 6, d, c)
                #x[:, [3,5], ...] = x[:, [5,3], ...]
                xcur = x_emb.clone().reshape(b, d, c)
                xl = torch.roll(x_emb.clone(), 1, 1).reshape(b, d, c)
                xr = torch.roll(x_emb.clone(), -1, 1).reshape(b, d, c)

                # xl = rearrange(xl, 'b (h w) c -> b h w c', h=self.hw[0], w=self.hw[1])
                # xl = xl[:, :, 3*self.hw[1]//4:, :]
                # xl = rearrange(xl, 'b h w c -> b (h w) c')
                # xr = rearrange(xr, 'b (h w) c -> b h w c', h=self.hw[0], w=self.hw[1])
                # xr = xr[:, :, :self.hw[1]//4, :]
                # xr = rearrange(xr, 'b h w c -> b (h w) c')

                xcur = self.multi_view_attn(xcur, xl, xr)
                x = xcur + x
                # x = x.reshape(b//6, 6, d, c)
                #x[:, [3,5], ...] = x[:, [5,3], ...]
                x = x.reshape(b, d, c)

        x = self.ff(self.norm3(x)) + x
        return x
    
    def position_embeding(self, img_feats_shape, img_metas, img_size=(256, 448), device='cuda'):
        eps = 1e-5
        pad_h, pad_w = img_size
        B, N, C, H, W = img_feats_shape
        coords_h = torch.arange(H, device=device).float() * pad_h / H
        coords_w = torch.arange(W, device=device).float() * pad_w / W

        index  = torch.arange(start=0, end=self.depth_num, step=1, device=device).float()
        bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
        coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        img2lidars = coords.new_tensor(img_metas) # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        with autocast(enabled=False):
            coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
            coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
            coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
            coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

            coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
            coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W)


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, use_multi_view_attn=False, use_local=False, with_position=False):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, 
                                   use_multi_view_attn=use_multi_view_attn, use_local=use_local, with_position=with_position)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, meta=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], meta=meta)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

