from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

# from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
from torch.utils import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

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


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward_plain(self, x):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)

    def forward(self, x, context=None, mask=None):
        if not XFORMERS_IS_AVAILBLE:
            return self.forward_plain(x)

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

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

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

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()
        
        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head) 
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, objs):

        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn( self.norm1(x), objs, objs)  
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) ) 
        
        return x 

class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs):

        N_visual = x.shape[1]
        objs = self.linear(objs)
        
        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x,objs],dim=1)))[:,0:N_visual,:]
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff(self.norm2(x) )  
        
        return x 

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, fuser_type=None):
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

        if fuser_type == "gatedSA":
            # note key_dim here actually is context_dim
            self.fuser = GatedSelfAttentionDense(dim, context_dim, n_heads, d_head) 
        # elif fuser_type == "gatedCA":
        #     self.fuser = GatedCrossAttentionDense(query_dim, key_dim, value_dim, n_heads, d_head) 
        else:
            assert False 

    def forward(self, x, context=None, objs=None):
        # return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        # if self.use_checkpoint and x.requires_grad:
        #     return checkpoint.checkpoint(self._forward, x, context, objs)
        # else:
        #     return self._forward(x, context, objs)
        return self._forward(x, context, objs)

    def _forward(self, x, context=None, objs=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        if objs is not None:
            x = self.fuser(x, objs)
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

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
                 use_checkpoint=True, fuser_type=None):
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
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, fuser_type=fuser_type)
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

    def forward(self, x, context=None, objs=None):
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
            x = block(x, context=context[i], objs=objs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

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

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
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

class DepthAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, output_bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Conv2d(query_dim, inner_dim, 1, 1, bias=False)
        self.to_k = nn.Conv3d(context_dim, inner_dim, 1, 1, bias=False)
        self.to_v = nn.Conv3d(context_dim, inner_dim, 1, 1, bias=False)
        if output_bias:
            self.to_out = nn.Conv2d(inner_dim, query_dim, 1, 1)
        else:
            self.to_out = nn.Conv2d(inner_dim, query_dim, 1, 1, bias=False)

    def forward(self, x, context):
        """

        @param x:        b,f0,h,w
        @param context:  b,f1,d,h,w
        @return:
        """
        hn, hd = self.heads, self.dim_head
        b, _, h, w = x.shape
        b, _, d, h, w = context.shape

        q = self.to_q(x).reshape(b,hn,hd,h,w) # b,t,h,w
        k = self.to_k(context).reshape(b,hn,hd,d,h,w) # b,t,d,h,w
        v = self.to_v(context).reshape(b,hn,hd,d,h,w) # b,t,d,h,w

        sim = torch.sum(q.unsqueeze(3) * k, 2) * self.scale # b,hn,d,h,w
        attn = sim.softmax(dim=2)

        # b,hn,hd,d,h,w * b,hn,1,d,h,w
        out = torch.sum(v * attn.unsqueeze(2), 3) # b,hn,hd,h,w
        out = out.reshape(b,hn*hd,h,w)
        
        return self.to_out(out)


class DepthTransformer(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, depth=None, checkpoint=True):
        super().__init__()
        inner_dim = n_heads * d_head
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1),
            nn.GroupNorm(8, inner_dim),
            nn.SiLU(True),
        )
        self.proj_context = nn.Sequential(
            nn.Conv3d(context_dim, context_dim, 1, 1, bias=False), # no bias
            nn.GroupNorm(8, context_dim),
            nn.ReLU(True), # only relu, because we want input is 0, output is 0
        )
        self.depth_attn = DepthAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, context_dim=context_dim, output_bias=False)  # is a self-attention if not self.disable_self_attn
        self.proj_out = nn.Sequential(
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            zero_module(nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)),
        )
        self.checkpoint = checkpoint
        # self.use_multi_view_attn = False
        # if type == 'multi_view_attn':
        #     self.use_multi_view_attn = True
        #     self.multi_view_attn = MultiViewCrossAttention(query_dim=dim, context_dim=context_dim,
        #                                     heads=n_heads, dim_head=d_head)
        if False:
            self.multi_view_attn = MultiViewCrossAttention(query_dim=inner_dim, context_dim=context_dim,
                                            heads=n_heads, dim_head=d_head)
        self.depth = depth
        self.pos_embed = zero_module(nn.Embedding(self.depth, context_dim))

    def forward(self, x, context=None):
        # return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        return self._forward(x, context)

    def _forward(self, x, context):
        x_in = x
        x = self.proj_in(x)
        context = self.proj_context(context)

        idx = torch.arange(self.depth, device=context.device).long().unsqueeze(0).unsqueeze(0)  # 1,1,D
        pos = self.pos_embed(idx).unsqueeze(0).permute(0,4,3,1,2)  # 1,1,1,D,C -> 1,C,D,1,1
        context = context + pos  # B,C,D,H,W + 1,C,D,1,1

        x = self.depth_attn(x, context)
        if False:
            if x.shape[0] == 6:
                # x[[3,5],...] = x[[5,3],...] 
                xcur = x.clone()
                xl = torch.roll(x.clone(), 1, 0)
                xr = torch.roll(x.clone(), -1, 0)
                B,C,H,W = xcur.shape
                xcur = xcur.reshape(B,C,H*W).permute(0,2,1)
                xl = xl.reshape(B,C,H*W).permute(0,2,1)
                xr = xr.reshape(B,C,H*W).permute(0,2,1)
                xcur = self.multi_view_attn(xcur, xl, xr).permute(0,2,1).reshape(B,C,H,W)
                x = xcur + x
                # x[[3,5],...] = x[[5,3],...] 
            elif x.shape[0] == 3:
                xcur,xl,xr = x[0,None].clone(),x[1,None].clone(),x[2,None].clone()
                B,C,H,W = xcur.shape
                xcur = xcur.reshape(B,C,H*W).permute(0,2,1)
                xl = xl.reshape(B,C,H*W).permute(0,2,1)
                xr = xr.reshape(B,C,H*W).permute(0,2,1)
                xcur = self.multi_view_attn(xcur, xl, xr).permute(0,2,1).reshape(B,C,H,W)
                x[0] = xcur + x[0]
    
        x = self.proj_out(x) + x_in
        return x