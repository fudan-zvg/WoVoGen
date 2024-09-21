import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from copy import deepcopy
from cldm.volume_transform import VolumeTransform

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, meta=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context, meta=meta)#, msk=msk[h.shape[-1]], msk_txt=msk_txt)
                hs.append(h)
            h = self.middle_block(h, emb, context, meta=meta) #, msk=msk[h.shape[-1]], msk_txt=msk_txt)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context, meta=meta)#, msk=msk[h.shape[-1]], msk_txt=msk_txt)

        h = h.type(x.dtype)
        return self.out(h)
    
    def get_multiviewatten_parameters(self):
        paras = []
        for name, para in self.input_blocks.named_parameters():
            name = name.split('.')
            if 'multi_view_attn' in name:
                paras.append(para)
        for name, para in self.middle_block.named_parameters():
            name = name.split('.')
            if 'multi_view_attn' in name:
                paras.append(para)
        return paras


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1), #, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1), #, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1), #, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, meta=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)#, msk[h.shape[-1]], msk_txt)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)#, msk[h.shape[-1]], msk_txt)
            outs.append(zero_conv(h, emb, context))
        h = self.middle_block(h, emb, context)#, msk[h.shape[-1]], msk_txt)
        outs.append(self.middle_block_out(h, emb, context))#, msk[h.shape[-1]], msk_txt))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, cfg_scale,
                 occ_encoder_config, view_num, sample_steps=50, 
                 origin_occ_shape=(16, 200, 200), 
                 input_size=(900, 1600), down_sample=8, use_multi_view_attn=False, grid_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self._init_schedule()

        self.cfg_scale = cfg_scale
        self.view_num = view_num
        self.occ_encoder = instantiate_from_config(occ_encoder_config)
        self.VT = VolumeTransform(with_DSE=True, 
                                  origin_occ_shape=origin_occ_shape, 
                                  input_size=input_size,
                                  down_sample=down_sample,
                                  grid_config=grid_config,
                                  )
        h, w = input_size
        latent_size = (h // down_sample, w // down_sample)
        self.sampler = SyncDDIMSampler(self, sample_steps , "uniform", 1.0, latent_size=latent_size)

        self.cross_view = use_multi_view_attn

    def _init_schedule(self):
        self.num_timesteps = 1000
        linear_start = 0.00085
        linear_end = 0.0120
        num_timesteps = 1000
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2 # T
        assert betas.shape[0] == self.num_timesteps

        # all in float64 first
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # T
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # T
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_log_variance_clipped = torch.clamp(posterior_log_variance_clipped, min=-10)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped.float())
    

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        batch_ = deepcopy(batch)
        B = batch_['jpg'].shape[0]
        N = self.view_num
        x = []
        c = []
        for i in range(N):
            batch_['jpg'] = batch['jpg'][torch.arange(B)[:,None],i][:, 0]
            batch_['txt'] = batch['txt'][i]
            x_, c_ = super().get_input(batch_, self.first_stage_key, *args, **kwargs)
            x.append(x_)
            c.append(c_)
        x = torch.stack(x, dim=1)
        c = torch.stack(c, dim=1)

        hint = batch_[self.control_key]
        hint = {i:hint[i].to(self.device) for i in hint}
        hint = {i:hint[i].to(memory_format=torch.contiguous_format).float() 
                for i in hint}

        return x, dict(c_crossattn=[c], c_concat=[hint])
    
    def apply_model(self, x_noisy, t, cond, volume_feats=None, *args, **kwargs):
        # assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        is_inference = False
        is_train = False
        if volume_feats is None:
            is_train = True
        else:
            is_inference = True
        

        if is_train:
            cond_txt = torch.cat(cond['c_crossattn'], 1)
            B = cond_txt.shape[0]
            N = self.view_num
            
            # occ encoder
            hint = cond['c_concat'][0]
            fts, pts = hint['fts'], hint['pts']
            occ_feat = self.occ_encoder(fts, pts)

            # import cv2
            # occ_feat_np = occ_feat['x'].detach().cpu().numpy()[0] #CDHW
            # occ_feat_np = occ_feat_np.transpose(2,3,1,0)
            # # occ_feat_np = occ_feat_np.transpose(1,2,0,3)
            # occ_feat_np = np.ascontiguousarray(occ_feat_np)
            # occ_feat_np = occ_feat_np.mean(-1).mean(-1)
            # occ_feat_np = (occ_feat_np - occ_feat_np.min()) / (occ_feat_np.max() - occ_feat_np.min())
            # occ_feat_np = (occ_feat_np * 255).astype(np.uint8)
            # cv2.imwrite('occ_Feat.jpg', occ_feat_np)

            if not self.cross_view:

                target_index = torch.randint(0, N, (B, 1), device=self.device).long()
                T = hint['T'][torch.arange(B)[:,None],target_index][:, 0] 
                K = hint['K'][torch.arange(B)[:,None],target_index][:, 0]
                volume_feats = self.VT(occ_feat['x'], K, T, target_index)

                # import cv2
                # volume_feats = self.VT(occ_feat['x'], hint['K'][0,1,None,...], hint['T'][0,1,None,...], target_index)
                # volume_feats_np = volume_feats.detach().cpu().numpy()[0] #CHW
                # volume_feats_np = volume_feats_np.transpose(1,2,0)
                # # occ_feat_np = occ_feat_np.transpose(1,2,0,3)
                # volume_feats_np = np.ascontiguousarray(volume_feats_np)
                # volume_feats_np = volume_feats_np.mean(-1)
                # volume_feats_np = (volume_feats_np - volume_feats_np.min()) / (volume_feats_np.max() - volume_feats_np.min())
                # volume_feats_np = (volume_feats_np * 255).astype(np.uint8)
                # cv2.imwrite('Volume_Feat.jpg', volume_feats_np)


                x_noisy_ = x_noisy[torch.arange(B)[:,None],target_index][:,0]
                cond_txt = cond_txt[torch.arange(B)[:,None],target_index][:,0]

                control = self.control_model(x=x_noisy_, hint=volume_feats, timesteps=t, context=cond_txt)  # 3,C,H,W
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                eps = diffusion_model(x=x_noisy_, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

                return eps, target_index

            if self.cross_view:
                cyc = lambda i:(i+N)%N
                t = t.repeat(3)
                target_index = torch.randint(0, 6, (B, 1), device=self.device)
                left_index, right_index = cyc(target_index - 1), cyc(target_index + 1)
                index = torch.cat([target_index, left_index, right_index], dim=-1)

                T, K = hint['T'], hint['K']

                # msk, msk_txt = hint['msk'], hint['msk_txt']
                # msk_txt = msk_txt.repeat(3,1,1,1)  # 3,6,77,1024
                # msk[:,[3,5],...] = msk[:,[5,3],...]

                 

                T = T[torch.arange(B)[:,None], index, ...].reshape(B*3, T.shape[2], T.shape[3])  # B,3,4,4
                K = K[torch.arange(B)[:,None], index, ...].reshape(B*3, K.shape[2], K.shape[3])  # B*3,3,3
                occ_feat = occ_feat['x']
                occ_feat = occ_feat.unsqueeze(1).repeat(1,3,1,1,1,1)\
                           .reshape(B*3, occ_feat.shape[1], occ_feat.shape[2], occ_feat.shape[3], occ_feat.shape[4])  # B*3,C,H,W,D
                volume_feats = self.VT(occ_feat, K, T, target_index)  # B*3,C,H,W

                img2ego = hint['img2ego']
                img2ego = img2ego[torch.arange(B)[:,None], index, ...].reshape(B*3, img2ego.shape[2], img2ego.shape[3])

                x_noisy_ = x_noisy[torch.arange(B)[:,None], index, ...].reshape(B*3, x_noisy.shape[2], x_noisy.shape[3], x_noisy.shape[4])  # B*3,4,H,W
                cond_txt = cond_txt[torch.arange(B)[:,None], index, ...].reshape(B*3, cond_txt.shape[2], cond_txt.shape[3])  # B*3,77,1024

                # msk = msk[torch.arange(B)[:,None], index, ...][:, 0]  # 3,6,H,W
                # msk.requires_grad = False
                # _, _, H, W = msk.shape
                # H, W = H // 8, W // 8
                # msk = [F.interpolate(msk, size=(H//2**i,W//2**i), mode='nearest') for i in range(0, 4)]
                # msk = {i.shape[-1]:i for i in msk}
                control = self.control_model(x=x_noisy_, hint=volume_feats, timesteps=t, context=cond_txt,
                                            #  msk=msk, msk_txt=msk_txt
                                             )  # 3,C,H,W
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                
                h, w = x_noisy_.shape[2:]
                hw = {(h//2**i) * (w//2**i):(h//2**i, w//2**i) for i in range(0,4)}
                meta = dict(img2ego=img2ego, is_train=is_train, hw = hw)

                eps = diffusion_model(x=x_noisy_, timesteps=t, context=cond_txt, meta=meta, 
                                      control=control,
                                    #   msk=msk, msk_txt=msk_txt, 
                                      only_mid_control=self.only_mid_control)
                eps = eps.reshape(B, 3, eps.shape[1], eps.shape[2], eps.shape[3])
                return eps[:, 0], target_index


        elif is_inference:
            control = self.control_model(x=x_noisy, hint=volume_feats, timesteps=t, context=cond,
                                        #  msk=kwargs['msk'], msk_txt=kwargs['msk_txt']
                                         )  # 3,C,H,W
            control = [c * scale for c, scale in zip(control, self.control_scales)]

            h, w = x_noisy.shape[2:]
            hw = {(h//2**i) * (w//2**i):(h//2**i, w//2**i) for i in range(0,4)}
            meta = dict(img2ego=kwargs['img2ego'], is_train=is_train, hw=hw)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, 
                                #   is_train=is_train,
                                  meta=meta,
                                  control=control, only_mid_control=self.only_mid_control,
                                #   msk=kwargs['msk'], msk_txt=kwargs['msk_txt']
                                  )
            return eps


    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    def sample(self, sampler, cond, cfg_scale, batch_view_num=1, return_inter_results=False, inter_interval=50, inter_view_interval=2):
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        hint = cond['c_concat'][0]
        fts, pts = hint['fts'], hint['pts']
        occ_feat = self.occ_encoder(fts, pts)

        K, T = hint['K'], hint['T']
        volume_feats = []
        for i in range(self.view_num):
            volume_feat = self.VT(occ_feat['x'][0, None], K[0,None, i], T[0,None, i])
            volume_feats.append(volume_feat)
        volume_feats = torch.cat(volume_feats, dim=0)
        img2ego = hint['img2ego'][0]

        # msk, msk_txt = hint['msk'], hint['msk_txt']
        # msk_txt = msk_txt.repeat(6,1,1,1)
        # B, N1, N2, H, W = msk.shape
        # msk = msk.reshape(B*N1, N2, H, W)
        # H, W = H // 8, W // 8
        # msk = [F.interpolate(msk, size=(H//2**i,W//2**i), mode='nearest') for i in range(0, 4)]
        # msk = {i.shape[-1]:i for i in msk}
        # volume_feats = {k:torch.stack([volume_feat[k] for volume_feat in volume_feats], dim=1) for k in volume_feat}

        x_sample, inter = sampler.sample(cond_txt, volume_feats, img2ego,
                                        #  msk=msk, msk_txt=msk_txt,
                                         unconditional_scale=cfg_scale, log_every_t=inter_interval, batch_view_num=batch_view_num)

        N = x_sample.shape[1]
        x_sample = torch.stack([self.decode_first_stage(x_sample[:, ni]) for ni in range(N)], 1)
        if return_inter_results:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            inter = torch.stack(inter['x_inter'], 2) # # B,N,T,C,H,W
            B,N,T,C,H,W = inter.shape
            inter_results = []
            for ni in tqdm(range(0, N, inter_view_interval)):
                inter_results_ = []
                for ti in range(T):
                    inter_results_.append(self.decode_first_stage(inter[:, ni, ti]))
                inter_results.append(torch.stack(inter_results_, 1)) # B,T,3,H,W
            inter_results = torch.stack(inter_results,1) # B,N,T,3,H,W
            return x_sample, inter_results
        else:
            return x_sample

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        # c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        # N = min(z.shape[0], N)
        # n_row = min(z.shape[0], n_row)
        # log["reconstruction"] = self.decode_first_stage(z)
        # log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key][0][0].split(',')[0], size=16)

        # recons_list = []
        # recons = [self.decode_first_stage(z[:,n]) for n in range(z.shape[1])]
        # for bi in range(1):
        #     img_pr_ = concat_images_list(*[recons[n][bi] for n in range(z.shape[1])])
        #     recons_list.append(img_pr_)
        # recons = torch.stack(recons_list, dim=0)
        # log["reconstruction"] = recons
        jpg_list = []
        for bi in range(1):
            img_pr_ = concat_images_list(*[batch['jpg'][bi][n] for n in range(6)])
            jpg_list.append(img_pr_)
        log["jpg"] = torch.stack(jpg_list, dim=0).permute(0,3,1,2)

        sample_list = []
        x_sample = self.sample(self.sampler, c, self.cfg_scale, self.view_num).permute(0, 1, 3, 4, 2)
        for bi in range(x_sample.shape[0]):
            img_pr_ = concat_images_list(*[x_sample[bi, n] for n in range(x_sample.shape[1])])
            sample_list.append(img_pr_) 
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = torch.stack(sample_list, dim=0).permute(0,3,1,2)

        # if plot_diffusion_rows:
        #     # get diffusion row
        #     diffusion_row = list()
        #     z_start = z[:n_row]
        #     for t in range(self.num_timesteps):
        #         if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
        #             t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
        #             t = t.to(self.device).long()
        #             noise = torch.randn_like(z_start)
        #             z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        #             diffusion_row.append(self.decode_first_stage(z_noisy))

        #     diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
        #     diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
        #     diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
        #     diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
        #     log["diffusion_row"] = diffusion_grid

        # if sample:
        #     # get denoise row
        #     samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
        #                                              batch_size=N, ddim=use_ddim,
        #                                              ddim_steps=ddim_steps, eta=ddim_eta)
        #     x_samples = self.decode_first_stage(samples)
        #     log["samples"] = x_samples
        #     if plot_denoise_rows:
        #         denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
        #         log["denoise_row"] = denoise_grid

        # if unconditional_guidance_scale > 1.0:
        #     uc_cross = self.get_unconditional_conditioning(N)
        #     uc_cat = c_cat  # torch.zeros_like(c_cat)
        #     uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        #     samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
        #                                      batch_size=N, ddim=use_ddim,
        #                                      ddim_steps=ddim_steps, eta=ddim_eta,
        #                                      unconditional_guidance_scale=unconditional_guidance_scale,
        #                                      unconditional_conditioning=uc_full,
        #                                      )
        #     x_samples_cfg = self.decode_first_stage(samples_cfg)
        #     log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    def q_sample(self, x_start, t):
        """
        @param x_start: B,*
        @param t:       B,
        @return:
        """
        B = x_start.shape[0]
        noise = torch.randn_like(x_start) # B,*

        sqrt_alphas_cumprod_  = self.sqrt_alphas_cumprod[t] # B,
        sqrt_one_minus_alphas_cumprod_ = self.sqrt_one_minus_alphas_cumprod[t] # B
        sqrt_alphas_cumprod_ = sqrt_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        sqrt_one_minus_alphas_cumprod_ = sqrt_one_minus_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        x_noisy = sqrt_alphas_cumprod_ * x_start + sqrt_one_minus_alphas_cumprod_ * noise
        return x_noisy, noise

    def p_losses(self, x_start, cond, t, noise=None):
        B = x_start.shape[0]
        x_noisy, noise = self.q_sample(x_start=x_start, t=t)
        model_output, index = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise[torch.arange(B)[:,None], index, ...][:, 0]
            # target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # loss_simple = self.get_loss(model_output.unsqueeze(0), target, mean=False).mean([1, 2, 3])
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # loss_vlb = self.get_loss(model_output.unsqueeze(0), target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def predict_with_unconditional_scale(self, x, t, clip_embed, volume_feats, unconditional_scale, img2ego=None,
                                         msk=None, msk_txt=None):
        # x_ = torch.cat([x] * 2, 0)
        # t_ = torch.cat([t] * 2, 0)
        clip_embed_ = self.get_unconditional_conditioning(x.shape[0])
        # clip_embed_ = torch.cat([clip_embed, clip_embed_], 0)

        # v_ = {}
        # for k, v in volume_feats.items():
        #     # v_[k] = torch.cat([v, torch.zeros_like(v)], 0)
        #     v_[k] = torch.zeros_like(v)
        # v_ = volume_feats
        # volume_feats = torch.cat([volume_feats, volume_feats], 0)

        # s, s_uc = self.apply_model(x_, t_, clip_embed_, volume_feats=volume_feats).chunk(2)

        s = self.apply_model(x, t, clip_embed, volume_feats=volume_feats, img2ego=img2ego,
                            #   msk=msk, msk_txt=msk_txt
                              )
        s_uc = self.apply_model(x, t, clip_embed_, volume_feats=volume_feats, img2ego=img2ego,
                                # msk=msk, msk_txt=msk_txt
                                )
        s = s_uc + unconditional_scale * (s - s_uc)
        return s

    def configure_optimizers(self):
        lr = self.learning_rate
        paras = []
        paras.append({"params": self.control_model.parameters(), "lr": lr},)
        paras.append({"params": self.occ_encoder.parameters(), "lr": lr},)
        paras.append({"params": self.VT.parameters(), "lr": lr},)
        # if not self.sd_locked:
        paras.append({"params": self.model.diffusion_model.output_blocks.parameters(), "lr": lr},)
        paras.append({"params": self.model.diffusion_model.out.parameters(), "lr": lr},)
        paras.append({"params": self.model.diffusion_model.get_multiviewatten_parameters(), "lr": lr},)
        
        
        opt = torch.optim.AdamW(paras, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

def concat_images_list(*args,vert=False):
    args = [i for i in args]
    row1 = args[0]
    row2 = args[3]
    for i in range(1,3):
        row1 = torch.concatenate([row1, args[i]], axis=1)
        row2 = torch.concatenate([row2, args[i+3]], axis=1)
    img = torch.concatenate([row1, row2], axis=0) 
    return img

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


class SyncDDIMSampler:
    def __init__(self, model: ControlLDM, ddim_num_steps, ddim_discretize="uniform", ddim_eta=1.0, latent_size=(112,200)):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.latent_size = latent_size
        self._make_schedule(ddim_num_steps, ddim_discretize, ddim_eta)
        self.eta = ddim_eta

    def _make_schedule(self,  ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose) # DT
        ddim_timesteps_ = torch.from_numpy(self.ddim_timesteps.astype(np.int64)) # DT

        alphas_cumprod = self.model.alphas_cumprod # T
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        self.ddim_alphas = alphas_cumprod[ddim_timesteps_].double() # DT
        self.ddim_alphas_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[ddim_timesteps_[:-1]]], 0) # DT
        self.ddim_sigmas = ddim_eta * torch.sqrt((1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev))

        self.ddim_alphas_raw = self.model.alphas[ddim_timesteps_].float() # DT
        self.ddim_sigmas = self.ddim_sigmas.float()
        self.ddim_alphas = self.ddim_alphas.float()
        self.ddim_alphas_prev = self.ddim_alphas_prev.float()
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas).float()

    @torch.no_grad()
    def denoise_apply_impl(self, x_target_noisy, index, noise_pred, is_step0=False):
        """
        @param x_target_noisy: B,N,4,H,W
        @param index:          index
        @param noise_pred:     B,N,4,H,W
        @param is_step0:       bool
        @return:
        """
        device = x_target_noisy.device
        B,N,_,H,W = x_target_noisy.shape

        # apply noise
        a_t = self.ddim_alphas[index].to(device).float().view(1,1,1,1,1)
        a_prev = self.ddim_alphas_prev[index].to(device).float().view(1,1,1,1,1)
        sqrt_one_minus_at = self.ddim_sqrt_one_minus_alphas[index].to(device).float().view(1,1,1,1,1)
        sigma_t = self.ddim_sigmas[index].to(device).float().view(1,1,1,1,1)

        pred_x0 = (x_target_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
        dir_xt = torch.clamp(1. - a_prev - sigma_t**2, min=1e-7).sqrt() * noise_pred
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        if not is_step0:
            noise = sigma_t * torch.randn_like(x_target_noisy)
            x_prev = x_prev + noise
        return x_prev

    @torch.no_grad()
    def denoise_apply(self, x_target_noisy, clip_embed, volume_feats, time_steps, index, unconditional_scale, img2ego=None,
                      msk=None, msk_txt=None, batch_view_num=1, is_step0=False):
        """
        @param x_target_noisy:   B,N,4,H,W
        @param input_info:
        @param clip_embed:       B,M,768
        @param time_steps:       B,
        @param index:            int
        @param unconditional_scale:
        @param batch_view_num:   int
        @param is_step0:         bool
        @return:
        """
        B, N, C, H, W = x_target_noisy.shape
        e_t = []
        B = 1
        x_target_noisy_ = x_target_noisy[:B].reshape(B*N, *x_target_noisy.shape[2:])
        clip_embed_ = clip_embed[:B].reshape(B*N, *clip_embed.shape[2:])
        time_steps_ = time_steps[:B].repeat(N)

        if unconditional_scale!=1.0:
            noise = self.model.predict_with_unconditional_scale(x_target_noisy_, time_steps_, clip_embed_, volume_feats, 
                                                                unconditional_scale, msk=msk, msk_txt=msk_txt, img2ego=img2ego)
        else:
            noise = self.model.model(x_target_noisy_, time_steps_, clip_embed_, volume_feats, is_train=False)
        e_t = noise.reshape(B, N, C, H, W)  # 1,N,4,H,W
        x_prev = self.denoise_apply_impl(x_target_noisy[:B], index, e_t, is_step0)
        return x_prev

    @torch.no_grad()
    def sample(self, context, volume_feats, img2ego=None, msk=None, msk_txt=None, unconditional_scale=1.0, log_every_t=50, batch_view_num=1):
        """
        @param input_info:      x, elevation
        @param clip_embed:      B,M,768
        @param unconditional_scale:
        @param log_every_t:
        @param batch_view_num:
        @return:
        """
        print(f"unconditional scale {unconditional_scale:.1f}")
        C, H, W = 4, self.latent_size[0], self.latent_size[1]
        N = self.model.view_num
        B = context.shape[0]
        
        device = self.model.device
        x_target_noisy = torch.randn([B, N, C, H, W], device=device)

        timesteps = self.ddim_timesteps
        intermediates = {'x_inter': []}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # index in ddim state
            time_steps = torch.full((B,), step, device=device, dtype=torch.long)
            x_target_noisy = self.denoise_apply(x_target_noisy, context, volume_feats, time_steps, index, unconditional_scale, img2ego=img2ego,
                                                msk=msk, msk_txt=msk_txt, batch_view_num=batch_view_num, is_step0=index==0)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_target_noisy)

        return x_target_noisy, intermediates
    