import einops
import torch
import torch as th
import torch.nn as nn
from copy import deepcopy
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
import numpy as np

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config

from cldm.attention import DepthTransformer
from cldm.volume_transform import LearnableVolumeTransform, VolumeTransform
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

class DecoderUnetModel(UNetModel):
    def __init__(self, volume_dims=(5,16,32,64), volume_depth=(48,36,24,12), *args, **kwargs):
        super().__init__(*args, **kwargs)
        for p in self.parameters():
            p.requires_grad = False
                
        model_channels = kwargs['model_channels']
        channel_mult = kwargs['channel_mult']
        d0,d1,d2,d3 = volume_dims
        depth0,depth1,depth2,depth3 = volume_depth

        self.input_conditions=nn.ModuleList()
        if True:
            self.input_b2c = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8}
            # self.input_b2c = {3:0,4:1,5:2,6:3,7:4,8:5,9:6}
            ch = model_channels*channel_mult[0]
            self.input_conditions.append(DepthTransformer(ch, 4, d0 // 2, context_dim=d0, depth=depth0))
            self.input_conditions.append(DepthTransformer(ch, 4, d0 // 2, context_dim=d0, depth=depth0))
            self.input_conditions.append(DepthTransformer(ch, 4, d1 // 2, context_dim=d1, depth=depth1))
            ch = model_channels*channel_mult[1]
            self.input_conditions.append(DepthTransformer(ch, 4, d1 // 2, context_dim=d1, depth=depth1))
            self.input_conditions.append(DepthTransformer(ch, 4, d1 // 2, context_dim=d1, depth=depth1))
            self.input_conditions.append(DepthTransformer(ch, 4, d2 // 2, context_dim=d2, depth=depth2))
            ch = model_channels*channel_mult[2]
            self.input_conditions.append(DepthTransformer(ch, 4, d2 // 2, context_dim=d2, depth=depth2))
            self.input_conditions.append(DepthTransformer(ch, 4, d2 // 2, context_dim=d2, depth=depth2))
            self.input_conditions.append(DepthTransformer(ch, 4, d3 // 2, context_dim=d3, depth=depth3))

        # 4
        ch = model_channels*channel_mult[2]
        self.middle_conditions = DepthTransformer(ch, 4, d3 // 2, context_dim=d3, depth=depth3)

        self.output_conditions=nn.ModuleList()
        self.output_b2c = {3:0,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8}
        # 8
        ch = model_channels*channel_mult[2]
        self.output_conditions.append(DepthTransformer(ch, 4, d2 // 2, context_dim=d2, depth=depth2)) # 0
        self.output_conditions.append(DepthTransformer(ch, 4, d2 // 2, context_dim=d2, depth=depth2)) # 1
        # # 16
        self.output_conditions.append(DepthTransformer(ch, 4, d1 // 2, context_dim=d1, depth=depth1)) # 2
        ch = model_channels*channel_mult[1]
        self.output_conditions.append(DepthTransformer(ch, 4, d1 // 2, context_dim=d1, depth=depth1)) # 3
        self.output_conditions.append(DepthTransformer(ch, 4, d1 // 2, context_dim=d1, depth=depth1)) # 4
        # 32
        self.output_conditions.append(DepthTransformer(ch, 4, d0 // 2, context_dim=d0, depth=depth0)) # 5
        ch = model_channels*channel_mult[0]
        self.output_conditions.append(DepthTransformer(ch, 4, d0 // 2, context_dim=d0, depth=depth0)) # 6
        self.output_conditions.append(DepthTransformer(ch, 4, d0 // 2, context_dim=d0, depth=depth0)) # 7
        self.output_conditions.append(DepthTransformer(ch, 4, d0 // 2, context_dim=d0, depth=depth0)) # 8

        # for para in self.input_blocks.parameters():
        #     para.requires_grad=False
        # for para in self.middle_block.parameters():
        #     para.requires_grad=False
        # for para in self.output_blocks.parameters():
        #     para.requires_grad=False
        
    def forward(self, x, timesteps=None, context=None, source=None, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for index, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            if index in self.input_b2c:
                layer = self.input_conditions[self.input_b2c[index]]
                h = layer(h, context=source[h.shape[-1]])
            hs.append(h)

        h = self.middle_block(h, emb, context)
        h = self.middle_conditions(h, context=source[h.shape[-1]])

        for index, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if index in self.output_b2c:
                layer = self.output_conditions[self.output_b2c[index]]
                h = layer(h, context=source[h.shape[-1]])

        h = h.type(x.dtype)
        return self.out(h)

    def get_depthattn_parameters(self):
        paras = [para for para in self.input_conditions.parameters()] +\
                [para for para in self.middle_conditions.parameters()] +\
                [para for para in self.output_conditions.parameters()]
        return paras

    def get_multiviewatten_parameters(self):
        paras = []
        for name, para in self.named_parameters():
            name = name.split('.')
            if 'multi_view_attn' in name:
                paras.append(para)
        return paras

    def get_unet_parameters(self):
        paras = [para for para in self.input_blocks.parameters()] +\
                [para for para in self.middle_block.parameters()] +\
                [para for para in self.output_blocks.parameters()]
        return paras

class DecoderLDM(LatentDiffusion):

    def __init__(self, control_key, occ_encoder_config, vt_config,
                 view_num, sample_steps=50, 
                 latent_size=(112,200), cfg_scale=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view_num = view_num
        self.control_key = control_key
        self.occ_encoder = instantiate_from_config(occ_encoder_config)
        # self.VT = instantiate_from_config(vt_config)
        self.VT = VolumeTransform(with_pos_emb=False)
        # self.LVT = LearnableVolumeTransform()
        self.cfg_scale = cfg_scale

        self._init_schedule()

        self.sampler = SyncDDIMSampler(self, sample_steps , "uniform", 1.0, latent_size=latent_size)
    
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

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)  # (1, 77, 1024)
        B = cond_txt.shape[0]
        N = self.view_num
        
        # occ encoder
        hint = cond['c_concat'][0]
        fts, pts = hint['fts'], hint['pts']
        occ_feat = self.occ_encoder(fts, pts)

        # get random view and project
        if False:
            t = t.repeat(3)
            cyc = lambda i:(i+N)%N
            target_index = torch.randint(0, 6, (B, 1), device=self.device)
            left_index, right_index = cyc(target_index - 1), cyc(target_index + 1)
            index = torch.cat([target_index, left_index, right_index]).squeeze(0)  # 3,1
            T = hint['T'][torch.arange(B)[:,None], index, ...][:, 0]
            K = hint['K'][torch.arange(B)[:,None], index, ...][:, 0]
            occ_feat = occ_feat['x'].repeat(3, 1, 1, 1, 1)
            volume_feat = self.LVT(occ_feat, K, T)
            volume_feats = {feat.shape[-1]:feat for feat in volume_feat}
            x_noisy_ = x_noisy[torch.arange(B)[:,None], index, ...][:, 0]
            cond_txt = cond_txt[torch.arange(B)[:,None], index, ...][:, 0]
        
        if False:
            t = t.repeat(6)
            T, K = hint['T'], hint['K']
            volume_feats = []
            for i in range(N):
                # volume_feat = self.VT.construct_view_frustum_volume(occ_feat['x'], K[:, i], T[:, i])
                volume_feat = self.LVT(occ_feat['x'], K[:, i], T[:, i])
                volume_feat = {feat.shape[-1]:feat for feat in volume_feat}
                volume_feats.append(volume_feat)
            volume_feats = {k:torch.cat([volume_feat[k] for volume_feat in volume_feats], dim=0) for k in volume_feat}
            t = t.repeat(6)
            x_noisy_ = x_noisy.reshape(B*N, *x_noisy.shape[2:])
            cond_txt = cond_txt.reshape(B*N, *cond_txt.shape[2:])
        # cyc = lambda i:(i+N)%N
        # for i in range(N):
            # view_cur, view_left, view_right = x_noisy[:, cyc(i)], x_noisy[:, cyc(i-1)], x_noisy[:, cyc(i+1)]
            # view_input = torch.cat([view_cur, view_left, view_right], dim=0)  # (3*1, 4, 112, 200)
            # txt_cur, txt_left, txt_right = cond_txt[:, cyc(i)], cond_txt[:, cyc(i-1)], cond_txt[:, cyc(i+1)]
            # txt_input = torch.cat([txt_cur, txt_left, txt_right], dim=0)  # (3*1, 77, 1024)
            # volume_cur, volume_left, volume_right = volume_feats[cyc(i)], volume_feats[cyc(i-1)], volume_feats[cyc(i+1)]
            # volume_input = {k:torch.cat([volume_cur[k], volume_left[k], volume_right[k]], dim=0) for k in volume_cur}
        target_index = torch.randint(0, N, (B, 1), device=self.device).long()
        T = hint['T'][torch.arange(B)[:,None],target_index][:, 0]
        K = hint['K'][torch.arange(B)[:,None],target_index][:, 0]
        volume_feats = self.VT(occ_feat['x'], K, T)
        # volume_feats = self.VT.construct_view_frustum_volume(occ_feat['x'], K, T)
        # volume_feat = self.LVT(occ_feat['x'], K, T)
        # volume_feats = {feat.shape[-1]:feat for feat in volume_feat}
        x_noisy_ = x_noisy[torch.arange(B)[:,None],target_index][:,0]
        cond_txt = cond_txt[torch.arange(B)[:,None],target_index][:,0]
        eps = diffusion_model(x=x_noisy_, timesteps=t, context=cond_txt, source=volume_feats)
        # eps = eps.reshape(3, *eps.shape[1:])[0,None]
        return eps, target_index

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
            target = noise[torch.arange(B)[:,None],index][:, 0]
            # target = noise[torch.arange(B)[:,None], index, ...][:, 0]
            # target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    def predict_with_unconditional_scale(self, x, t, clip_embed, volume_feats, unconditional_scale):
        # x_ = torch.cat([x] * 2, 0)
        # t_ = torch.cat([t] * 2, 0)
        clip_embed_ = self.get_unconditional_conditioning(x.shape[0])
        # clip_embed_ = torch.cat([clip_embed, clip_embed_], 0)

        v_ = {}
        for k, v in volume_feats.items():
            # v_[k] = torch.cat([v, torch.zeros_like(v)], 0)
            v_[k] = torch.zeros_like(v)

        # s, s_uc = self.model.diffusion_model(x_, t_, clip_embed_, source=v_).chunk(2)
        s = self.model.diffusion_model(x, t, clip_embed, source=volume_feats)
        s_uc = self.model.diffusion_model(x, t, clip_embed_, source=v_)
        s = s_uc + unconditional_scale * (s - s_uc)
        return s
    
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2,ddim_steps=50, unconditional_guidance_scale=9.0, **kwargs):

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        recons_list = []
        recons = [self.decode_first_stage(z[:,n]) for n in range(z.shape[1])]
        for bi in range(1):
            img_pr_ = concat_images_list(*[recons[n][bi] for n in range(z.shape[1])])
            recons_list.append(img_pr_)
        recons = torch.stack(recons_list, dim=0)
        log["reconstruction"] = recons
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        sample_list = []
        x_sample = self.sample(self.sampler, batch, self.cfg_scale, self.view_num)
        for bi in range(x_sample.shape[0]):
            img_pr_ = concat_images_list(*[x_sample[bi, n] for n in range(x_sample.shape[1])])
            sample_list.append(img_pr_)
        x_sample = torch.stack(sample_list, dim=0)        
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_sample
        
        return log

    def sample(self, sampler, batch, cfg_scale, batch_view_num, return_inter_results=False, inter_interval=50, inter_view_interval=2):
        c = []
        for i in range(self.view_num):
            c_ = self.get_learned_conditioning(batch['txt'])
            c.append(c_)
        c = torch.stack(c, dim=1)

        hint = batch['hint']
        hint = {i:hint[i].to(self.device) for i in hint}
        hint = {i:hint[i].to(memory_format=torch.contiguous_format).float() 
                for i in hint}
        fts, pts = hint['fts'], hint['pts']
        occ_feat = self.occ_encoder(fts, pts)

        K, T = hint['K'], hint['T']
        volume_feats = []
        for i in range(self.view_num):
            volume_feat = self.VT(occ_feat['x'], K[:, i], T[:, i])
            # volume_feat = self.VT.construct_view_frustum_volume(occ_feat['x'], K[:, i], T[:, i])
            # volume_feat = self.LVT(occ_feat['x'], K[:, i], T[:, i])
            # volume_feat = {feat.shape[-1]:feat for feat in volume_feat}
            volume_feats.append(volume_feat)
        volume_feats = {k:torch.stack([volume_feat[k] for volume_feat in volume_feats], dim=1) for k in volume_feat}
        # volume_feats = {k:torch.cat([volume_feat[k] for volume_feat in volume_feats], dim=0) for k in volume_feat}
        # volume_feats = None
        # for i in range(self.view_num):
        #     volume_feat = self.VT.construct_view_frustum_volume(occ_feat['x'], K[:,i], T[:,i])
        #     for k in volume_feat:
        #         volume_feat[k] = volume_feat[k].unsqueeze(1)
        #     if volume_feats == None:
        #         volume_feats = volume_feat
        #     else:
        #         for k in volume_feat:
        #             volume_feats[k] = torch.cat([volume_feats[k], volume_feat[k]], dim=1)

        x_sample, inter = sampler.sample(c, volume_feats, unconditional_scale=cfg_scale, log_every_t=inter_interval, batch_view_num=batch_view_num)

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
        
    def configure_optimizers(self):
        lr = self.learning_rate
        paras = []
        # paras.append({"params": self.model.diffusion_model.parameters(), "lr": lr},)
        # paras.append({"params": self.model.diffusion_model.get_unet_parameters(), "lr": lr},)
        paras.append({"params": self.model.diffusion_model.get_depthattn_parameters(), "lr": lr},)
        # paras.append({"params": self.model.diffusion_model.get_multiviewatten_parameters(), "lr": lr},)
        paras.append({"params": self.occ_encoder.parameters(), "lr": lr},)
        paras.append({"params": self.VT.parameters(), "lr": lr},)
        # paras.append({"params": self.LVT.parameters(), "lr": lr},)
        # paras.append({"params": self.cond_stage_model.parameters(), "lr": lr},)
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
    row1 = args[0]
    row2 = args[3]
    for i in range(1,3):
        row1 = torch.concatenate([row1, args[i]], axis=2)
        row2 = torch.concatenate([row2, args[i+3]], axis=2)
    img = torch.concatenate([row1, row2], axis=1) 
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

def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

class SyncDDIMSampler:
    def __init__(self, model: DecoderLDM, ddim_num_steps, ddim_discretize="uniform", ddim_eta=1.0, latent_size=(112,200)):
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
    def denoise_apply(self, x_target_noisy, clip_embed, volume_feats, time_steps, index, unconditional_scale, batch_view_num=1, is_step0=False):
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
        volume_feats_ = {i:volume_feats[i][:B] for i in volume_feats}
        volume_feats = {k:volume_feats_[k].reshape(B*N, *volume_feats_[k].shape[2:]) for k in volume_feats_}
        

        # for ni in range(0, N):
        #     x_target_noisy_ = x_target_noisy[:B, ni]
        #     clip_embed_ = clip_embed[:B, ni]
        #     volume_feats_ = {i:volume_feats[i][:B, ni] for i in volume_feats}
        #     if unconditional_scale!=1.0:
        #         noise = self.model.predict_with_unconditional_scale(x_target_noisy_, time_steps_, clip_embed_, volume_feats_, unconditional_scale)
        #     else:
        #         noise = self.model.model(x_target_noisy_, time_steps_, clip_embed_, volume_feats_, is_train=False)
        #     e_t.append(noise.view(B,4,H,W))
        if unconditional_scale!=1.0:
            noise = self.model.predict_with_unconditional_scale(x_target_noisy_, time_steps_, clip_embed_, volume_feats, unconditional_scale)
        else:
            noise = self.model.model(x_target_noisy_, time_steps_, clip_embed_, volume_feats, is_train=False)
        e_t = noise.reshape(B, N, C, H, W)  # 1,N,4,H,W
        x_prev = self.denoise_apply_impl(x_target_noisy[:B], index, e_t, is_step0)
        return x_prev

    @torch.no_grad()
    def sample(self, context, volume_feats, unconditional_scale=1.0, log_every_t=50, batch_view_num=1):
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
            x_target_noisy = self.denoise_apply(x_target_noisy, context, volume_feats, time_steps, index, unconditional_scale, batch_view_num=batch_view_num, is_step0=index==0)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_target_noisy)

        return x_target_noisy, intermediates