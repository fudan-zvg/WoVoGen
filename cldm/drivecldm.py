import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

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

class ControlCatLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, cfg_scale,
                 occ_encoder_config, view_num, msk_num, sample_steps=50, use_local=False,
                 origin_occ_shape=(16, 200, 200), padding=0,
                 input_size=(900, 1600), down_sample=8, use_multi_view_attn=False, grid_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self._init_schedule()

        self.cfg_scale = cfg_scale
        self.view_num = view_num
        self.msk_num = msk_num
        self.occ_encoder = instantiate_from_config(occ_encoder_config)
        self.VT = VolumeTransform(with_DSE=True, 
                                  origin_occ_shape=origin_occ_shape, 
                                  input_size=input_size,
                                  down_sample=down_sample,
                                  grid_config=grid_config,
                                  )
        h, w = input_size

        self.padding = padding
        self.feat_padding = self.padding // down_sample
        latent_size = (2*h // down_sample + 3*self.feat_padding, 3*w // down_sample + 4*self.feat_padding)
        self.sampler = SyncCatDDIMSampler(self, sample_steps , "uniform", 1.0, latent_size=latent_size)

        self.cross_view = use_multi_view_attn
        self.use_local = use_local

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
    
    @staticmethod
    def concat_image(tensor: torch.Tensor) -> torch.Tensor:
        # B N C H W
        B, N, C, H, W = tensor.shape

        # Split the tensor along the N dimension
        tensors = tensor.split(1, dim=1)  # This will be a list of 6 tensors each of shape [B, 1, C, H, W]

        # Now, we'll remove the single-dimensional entries from the tensors
        tensors = [t.squeeze(1) for t in tensors]  # Now each tensor has shape [B, C, H, W]

        # We need to form two rows, top (first 3 tensors) and bottom (last 3 tensors)
        top_row = torch.cat(tensors[:3], dim=3)  # Concatenate along W dimension, resulting in shape [B, C, H, 3W]
        bottom_row = torch.cat(tensors[3:], dim=3)  # Also [B, C, H, 3W]

        # Finally, concatenate top and bottom rows along H dimension
        result = torch.cat([top_row, bottom_row], dim=2)  # Resulting shape [B, C, 2H, 3W]
        return result

    
    def apply_model(self, x_noisy, t, cond, volume_feats=None, meta=dict(), *args, **kwargs):
        # assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
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

            T = hint['T'].flatten(0, 1)
            K = hint['K'].flatten(0, 1)
            B_oc, C_oc, D_oc, H_oc, W_oc = occ_feat['x'].shape
            occ_repeat = occ_feat['x'].unsqueeze(1).expand(-1, N, -1, -1, -1, -1).flatten(0,1)
            volume_feat = rearrange(self.VT(occ_repeat, K, T), '(b n) c h w -> b n c h w', b=B, n=N)

            volume_feats_cat = self.concat_image(volume_feat)
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

            x_noisy_cat = self.concat_image(x_noisy)
            
            cond_txt = cond_txt[:, 0, :, :]
            control = self.control_model(x=x_noisy_cat, hint=volume_feats_cat, timesteps=t, context=cond_txt)  # 3,C,H,W
            control = [c * scale for c, scale in zip(control, self.control_scales)]

            meta['is_train'] = self.training
            h, w = x_noisy_cat.shape[2:]
            hw = {(h//2**i) * (w//2**i):(h//2**i, w//2**i) for i in range(0,4)}
            meta['hw'] = hw
            if self.use_local:
                msk, msk_txt = hint['msk'], hint['msk_txt']
                msk = rearrange(msk, 'b n1 n2 h w -> (b n2) n1 1 h w')
                msk_cat = self.concat_image(msk)
                msk_cat = [F.interpolate(msk_cat, size=(h//2**i,w//2**i), mode='nearest') for i in range(0, 4)]
                msk_cat = {i.shape[-1]:rearrange(i, '(b n2) 1 h w -> b n2 h w', b=hint['msk'].shape[0], n2=self.msk_num)
                            for i in msk_cat}
                meta['msk'], meta['msk_txt'] = msk_cat, msk_txt

            eps = diffusion_model(x=x_noisy_cat, timesteps=t, meta=meta, 
                                  context=cond_txt, control=control, only_mid_control=self.only_mid_control)

            return eps, None

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


    def test_apply_model(self, noisy, t, cond, volume_feats=None, meta=dict(), *args, **kwargs):
        # assert isinstance(cond, dict)
        volume_feats_cat = self.concat_image(volume_feats.unsqueeze(0))
        cond = cond[0].unsqueeze(0)
        diffusion_model = self.model.diffusion_model
        
        control = self.control_model(x=noisy, hint=volume_feats_cat, timesteps=t, context=cond,
                                    #  msk=kwargs['msk'], msk_txt=kwargs['msk_txt']
                                        )  # 3,C,H,W
        control = [c * scale for c, scale in zip(control, self.control_scales)]

        # h, w = noisy.shape[2:]
        # hw = {(h//2**i) * (w//2**i):(h//2**i, w//2**i) for i in range(0,4)}
        # meta = dict(img2ego=kwargs['img2ego'], is_train=False, hw=hw)
        N = self.view_num
        meta_ = deepcopy(meta)
        h, w = volume_feats_cat.shape[2:]
        hw = {(h//2**i) * (w//2**i):(h//2**i, w//2**i) for i in range(0,4)}
        meta_['hw'] = hw
        if self.use_local:
            msk, msk_txt = meta_['msk'][0, None], meta_['msk_txt'][0, None]
            msk = rearrange(msk, 'b n1 n2 h w -> (b n2) n1 1 h w')
            msk_cat = self.concat_image(msk)
            msk_cat = [F.interpolate(msk_cat, size=(h//2**i,w//2**i), mode='nearest') for i in range(0, 4)]
            msk_cat = {i.shape[-1]:rearrange(i, '(b n2) 1 h w -> b n2 h w', b=volume_feats_cat.shape[0], n2=self.msk_num)
                        for i in msk_cat}
            meta_['msk'], meta_['msk_txt'] = msk_cat, msk_txt
        eps = diffusion_model(x=noisy, timesteps=t, context=cond, 
                                meta=meta_,
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

        B, N = hint['K'].shape[:2]
        occ_repeat = occ_feat['x'].unsqueeze(1).expand(-1, N, -1, -1, -1, -1).flatten(0,1)
        T = hint['T'].flatten(0, 1)
        K = hint['K'].flatten(0, 1)
        volume_feats = rearrange(self.VT(occ_repeat, K, T), '(b n) c h w -> b n c h w', b=B, n=N)
   

        # msk, msk_txt = hint['msk'], hint['msk_txt']
        # msk_txt = msk_txt.repeat(6,1,1,1)
        # B, N1, N2, H, W = msk.shape
        # msk = msk.reshape(B*N1, N2, H, W)
        # H, W = H // 8, W // 8
        # msk = [F.interpolate(msk, size=(H//2**i,W//2**i), mode='nearest') for i in range(0, 4)]
        # msk = {i.shape[-1]:i for i in msk}
        # volume_feats = {k:torch.stack([volume_feat[k] for volume_feat in volume_feats], dim=1) for k in volume_feat}
        meta = dict(msk=hint['msk'], msk_txt=hint['msk_txt'], img2ego=hint['img2ego'], is_train=False)
        x_sample, inter = sampler.sample(cond_txt, volume_feats, meta=meta,
                                         unconditional_scale=cfg_scale, log_every_t=inter_interval, batch_view_num=batch_view_num)

        x_sample = self.decode_first_stage(x_sample)
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

        jpg_list = []
        for bi in range(1):
            img_pr_ = concat_images_list(*[batch['jpg'][bi][n] for n in range(self.view_num)])
            jpg_list.append(img_pr_)
        log["jpg"] = torch.stack(jpg_list, dim=0).permute(0,3,1,2)

        if self.use_local:
            msk_list = []
            for i in range(self.view_num):
                for bi in range(1):
                    img_pr_ = concat_images_list(*[batch['hint']['msk'][bi][n][i] for n in range(self.msk_num)])
                    msk_list.append(img_pr_)
            log["msk"] = msk_list  #.permute(0,3,1,2)

        x_sample = self.sample(self.sampler, c, self.cfg_scale, self.view_num)
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_sample


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
        noise_cat = self.concat_image(noise)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            # target = noise[torch.arange(B)[:,None], index, ...][:, 0]
            target = noise_cat
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

    def predict_with_unconditional_scale(self, x, t, clip_embed, volume_feats, unconditional_scale, meta=None):
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

        s = self.test_apply_model(x, t, clip_embed, volume_feats=volume_feats, meta=meta
                              )
        s_uc = self.test_apply_model(x, t, clip_embed_, volume_feats=volume_feats, meta=meta
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



class SyncCatDDIMSampler:
    def __init__(self, model: ControlCatLDM, ddim_num_steps, ddim_discretize="uniform", ddim_eta=1.0, latent_size=(112,200)):
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
        B,_,H,W = x_target_noisy.shape

        # apply noise
        a_t = self.ddim_alphas[index].to(device).float().view(1,1,1,1)
        a_prev = self.ddim_alphas_prev[index].to(device).float().view(1,1,1,1)
        sqrt_one_minus_at = self.ddim_sqrt_one_minus_alphas[index].to(device).float().view(1,1,1,1)
        sigma_t = self.ddim_sigmas[index].to(device).float().view(1,1,1,1)

        pred_x0 = (x_target_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
        dir_xt = torch.clamp(1. - a_prev - sigma_t**2, min=1e-7).sqrt() * noise_pred
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        if not is_step0:
            noise = sigma_t * torch.randn_like(x_target_noisy)
            x_prev = x_prev + noise
        return x_prev

    @torch.no_grad()
    def denoise_apply(self, x_target_noisy, clip_embed, volume_feats, time_steps, index, unconditional_scale, meta=None, is_step0=False):
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
        B, C, H, W = x_target_noisy.shape
        e_t = []
        B = 1
        N=6
        x_target_noisy_ = x_target_noisy[:B].reshape(B, *x_target_noisy.shape[1:])
        clip_embed_ = clip_embed[:B].reshape(B*N, *clip_embed.shape[2:])
        time_steps_ = time_steps[:B]
        volume_feats = volume_feats[:B].reshape(B*N, *volume_feats.shape[2:])

        if unconditional_scale!=1.0:
            noise = self.model.predict_with_unconditional_scale(x_target_noisy_, time_steps_, clip_embed_, volume_feats, 
                                                                unconditional_scale, meta=meta)
        else:
            noise = self.model.model(x_target_noisy_, time_steps_, clip_embed_, volume_feats, is_train=False)
        e_t = noise.reshape(B, C, H, W)  # 1,N,4,H,W
        x_prev = self.denoise_apply_impl(x_target_noisy[:B], index, e_t, is_step0)
        return x_prev

    @torch.no_grad()
    def sample(self, context, volume_feats, meta=None, unconditional_scale=1.0, log_every_t=50, batch_view_num=1):
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
        x_target_noisy = torch.randn([B, C, H, W], device=device)

        timesteps = self.ddim_timesteps
        intermediates = {'x_inter': []}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # index in ddim state
            time_steps = torch.full((B,), step, device=device, dtype=torch.long)
            x_target_noisy = self.denoise_apply(x_target_noisy, context, volume_feats,
                                                time_steps, index, unconditional_scale, meta=meta, is_step0=index==0)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_target_noisy)

        return x_target_noisy, intermediates