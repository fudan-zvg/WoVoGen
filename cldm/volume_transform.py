import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import spconv.pytorch as spconv
from cldm.spconv_blocks import post_act_block, SparseBasicBlock
from .utils_3d import get_range
from einops import rearrange, repeat
from torch.cuda.amp import autocast, GradScaler

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_sizes[0])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        self.hidden_to_output = nn.Linear(hidden_sizes[-1], output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.input_to_hidden(x)
        x = self.activation(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation(x)
        x = self.hidden_to_output(x)
        return x

class LearnableVolumeTransform(nn.Module):
    def __init__(self, input_channel=64, base_channel=128, out_channel=1024, norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super(LearnableVolumeTransform, self).__init__()
        self.basic_frustum_size = [48, 112, 200]
        self.sparse_shape_xyz = self.basic_frustum_size[1:] + [self.basic_frustum_size[0]]
        self.origin_occ_shape = [200, 200, 16]
        self.origin_camera_size = [900, 1600]
        self.origin_occ_size = 0.4
        self.origin_D_length = 6.4
        
        block = post_act_block
        
        self.conv0 = spconv.SparseSequential(
            block(input_channel, base_channel, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv0', conv_type='spconv'),
            SparseBasicBlock(base_channel, base_channel, norm_cfg=norm_cfg, indice_key='res0'),
            SparseBasicBlock(base_channel, base_channel, norm_cfg=norm_cfg, indice_key='res0'),
        )

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
        )

    def get_sampling_points(self, target_size, maximum_depth):
        ds, hs, ws = target_size[0], target_size[1], target_size[2]
        W, H = self.origin_camera_size[1], self.origin_camera_size[0]
        x_coords = torch.linspace(0, H - 1, hs)
        y_coords = torch.linspace(0, W - 1, ws)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
        sampled_points = torch.stack((grid_y, grid_x), dim=2).unsqueeze(2).repeat(1, 1, ds, 1) #(hs, ws, ds, 2)-(xy)
        
        depth = torch.ones((sampled_points.shape[0], sampled_points.shape[1], ds, 1)) # (hs, ws, ds, 1)
        depth_range = torch.linspace(0.5, maximum_depth, ds).unsqueeze(0).unsqueeze(0).repeat(hs, ws, 1).unsqueeze(-1) #(hs, ws, ds, 1)
        depth = depth * depth_range
        
        sampled_points = sampled_points * depth_range.repeat(1, 1, 1, 2)
        sampled_points = torch.cat([sampled_points, depth], axis=-1) # (hs, ws, ds, 3)-(xyz)
        sampled_points = sampled_points.permute(2, 0, 1, 3) # (ds, hs, ws, 3)
        
        x_coords_sp = torch.linspace(0, ds, ds)
        y_coords_sp = torch.linspace(0, ws, ws)
        z_coords_sp = torch.linspace(0, hs, hs)
        grid_x_sp, grid_y_sp, grid_z_sp = torch.meshgrid(x_coords_sp, y_coords_sp, z_coords_sp)
        sampled_points_sp = torch.stack((grid_x_sp, grid_y_sp, grid_z_sp), dim=3)
        
        return sampled_points, sampled_points_sp
        
    def back_projection(self, sampled_points, K, T):
        D, H, W = sampled_points.shape[0], sampled_points.shape[1], sampled_points.shape[2]#dhw in camera
        sampled_points = sampled_points.flatten(0, 2)
        normalized_points = (torch.inverse(K) @ (sampled_points.T/100.)).T*100.
        
        rotation, translation = T[:3, :3], T[:3, 3]
        homogeneous_3d_points = normalized_points @ rotation.T + translation
        
        homogeneous_3d_points = homogeneous_3d_points.reshape(D, H, W, 3) #?check
        return homogeneous_3d_points
        
    def construct_view_frustum_volume(self, occ_feature, K, T):
        C, H_occ, W_occ, D_occ = occ_feature.shape[0], occ_feature.shape[1], occ_feature.shape[2], occ_feature.shape[3]
        ratio = self.origin_occ_shape[0] // H_occ
        occ_size = self.origin_occ_size * ratio
        bias = torch.FloatTensor([(H_occ - 1) / 2, (W_occ - 1) / 2, (D_occ - 1) / self.origin_D_length]).to(occ_feature.device)

        sampled_points, sampled_points_sp = self.get_sampling_points(self.basic_frustum_size, H_occ / 2 - 0.5) #(ds, hs, ws, 3)
        sampled_points = sampled_points.to(occ_feature.device)
        sampled_points_sp = sampled_points_sp.to(occ_feature.device)
        points_in_threeD = self.back_projection(sampled_points, K, T)
        points_in_threeD = points_in_threeD + bias
                
        points_in_threeD[:,:,:,0] = points_in_threeD[:,:,:,0] / (H_occ / 2) - 1.
        points_in_threeD[:,:,:,1] = points_in_threeD[:,:,:,1] / (W_occ / 2) - 1.
        points_in_threeD[:,:,:,2] = points_in_threeD[:,:,:,2] / (D_occ / 2) - 1.
                
        sampled_volume = F.grid_sample(occ_feature.unsqueeze(0), points_in_threeD.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True) #(1, C, ds, hs, ws)
        return sampled_volume.squeeze().permute(1, 2, 3, 0), sampled_points_sp #(ds, hs, ws, C)
        
    def construct_sparse_voxel(self, voxel_features, corrs):
        B, D = voxel_features.shape[0], voxel_features.shape[-1]
        voxel_features = voxel_features.view(-1, D)
        N = voxel_features.shape[0]
        
        corrs = corrs.view(-1, 3)[:, [2, 1, 0]] # change hwd-xyz to zyx-dwh
        batch_tensor = torch.arange(B).unsqueeze(-1).repeat(1, N // B).view(-1).unsqueeze(-1).to(voxel_features.device)
        corrs = torch.cat([batch_tensor, corrs], axis=1)
        return voxel_features, corrs.int()
        
    def forward(self, occ_feature, K, T):
        '''
        occ_feature: (B, C, H, W, D) hwd in occ - xyz in 3d
        K: (B, 3, 3)
        T: (B, 4, 4)
        '''
        B = occ_feature.shape[0]
        volume_feature, points_corrs = None, None 
        for idb in range(B):
            single_volume_feature, single_points_corrs = self.construct_view_frustum_volume(occ_feature[idb], K[idb], T[idb]) #(D, H, W, C), (D, H, W, 3)
            if volume_feature is None and points_corrs is None:
                volume_feature = single_volume_feature.unsqueeze(0)
                points_corrs = single_points_corrs.unsqueeze(0)
            else:
                volume_feature = torch.cat([volume_feature, single_volume_feature.unsqueeze(0)], axis=0) # (B, D, H, W, C)
                points_corrs = torch.cat([points_corrs, single_points_corrs.unsqueeze(0)], axis=0) # (B, D, H, W, 3)
            
        voxel_features, corrs = self.construct_sparse_voxel(volume_feature, points_corrs)
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, corrs, self.sparse_shape_xyz[::-1], B)
            
        x_conv0 = self.conv0(input_sp_tensor) # dense [2, 64, 112, 200, 48] -> [2, 128, 56, 100, 24]
        x_conv1 = self.conv1(x_conv0) # dense [2, 128, 56, 100, 24] -> [2, 256, 28, 50, 12]
        x_conv2 = self.conv2(x_conv1) # dense [2, 256, 28, 50, 12] -> [2, 512, 14, 25, 6]
        
        out1 = torch.sum(volume_feature.permute(0,4,1,2,3), dim=2)
        out2 = torch.sum(x_conv0.dense().permute(0,1,2,4,3), dim=2)
        out3 = torch.sum(x_conv1.dense().permute(0,1,2,4,3), dim=2)
        out4 = torch.sum(x_conv2.dense().permute(0,1,2,4,3), dim=2)
        #print(out1[out1!=0].shape, out2[out2!=0].shape, out3[out3!=0].shape, out4[out4!=0].shape)
        return out1, out2, out3, out4
        
class DSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        B, C, D, H, W = x.size() #(b, c, ds, hs, ws)
        x = x.permute(0, 2, 1, 3, 4) # (b, ds, c, hs, ws)
        y = self.avg_pool(x).view(B, D) #(b, ds)
        y = self.fc(y).view(B, D, 1, 1, 1) #(b, ds, 1, 1, 1) 
        x = x * y.expand_as(x)
        return x.permute(0, 2, 1, 3, 4)
    

class VolumeTransform(nn.Module):
    def __init__(self, with_DSE=False, origin_occ_shape=(16,200, 200), input_size=(900, 1600), down_sample=8, grid_config=None):
        super(VolumeTransform, self).__init__()
        h, w = input_size
        # self.target_frustum_size = [48, h//down_sample, w//down_sample]
        self.origin_occ_shape = origin_occ_shape
        self.origin_camera_size = input_size
        self.down_sample = down_sample
        self.origin_occ_size = 0.4
        self.origin_D_length = 6.4
        self.grid_config = grid_config
        
        self.with_DSE = with_DSE
        if self.with_DSE:
            d_bound = self.grid_config['d_bound']
            fd = int((d_bound[1] - d_bound[0]) // d_bound[2])
            self.DSE = DSELayer(channel=fd)    
        
    def get_sampling_points(self):
        ih, iw = self.origin_camera_size
        fh, fw = ih // self.down_sample, iw // self.down_sample
        d_bound = self.grid_config['d_bound']
        fd = int((d_bound[1] - d_bound[0]) // d_bound[2])
        x_coords = torch.linspace(0, iw - 1, fw)
        y_coords = torch.linspace(0, ih - 1, fh)
        d_coords = torch.linspace(d_bound[0], d_bound[1], fd)
        grid_d, grid_y, grid_x = torch.meshgrid(d_coords, y_coords, x_coords)
        sampled_points = torch.stack((grid_x, grid_y, grid_d), dim=-1)

        sampled_points = torch.cat([sampled_points[..., :2] * sampled_points[..., 2:3], 
                                    sampled_points[..., 2:3], torch.ones_like(sampled_points[..., 2:3])], dim=-1) #(ws, hs, ds, 4)
        return sampled_points
        
    def back_projection(self, volume_points, K, T):
        B, D, H, W, _ = volume_points.shape # BDHW4
        cam2imgs = repeat(torch.eye(4).to(K), 'p q -> bs p q', bs=B)
        cam2imgs[:, :3, :3] = K
        # cam_points = rearrange(torch.inverse(cam2imgs), 'bs p q -> bs 1 1 1 p q') @ (volume_points/10).unsqueeze(-1)
        
        cam_points = rearrange(torch.inverse(cam2imgs), 'bs p q -> bs 1 1 1 p q') @ (volume_points).unsqueeze(-1)
            # cam_points = cam_points*10
        ego_points = rearrange(T, 'bs p q -> bs 1 1 1 p q') @ cam_points
        return ego_points
        
    def forward(self, occ_feature, K, T, idx=None):
        '''
        occ_feature: (B, C, D, H, W) dhw in occ
        K: (B, 3, 3)
        T: (B, 4, 4)
        '''
        dx, bx, nx, pc_range = get_range(self.grid_config)
        B, C, D_occ, H_occ, W_occ,  = occ_feature.shape
        D_oi, H_oi, W_oi = self.origin_occ_shape
        ratio = (W_oi / W_occ, H_oi / H_occ, D_oi / D_occ)
        with autocast(enabled=False):
            ego2egofeat = torch.eye(4)
            ego2egofeat[0,0] = 1 / self.grid_config['x_bound'][2] / ratio[0]
            ego2egofeat[1,1] = 1 / self.grid_config['y_bound'][2] / ratio[1]
            ego2egofeat[2,2] = 1 / self.grid_config['z_bound'][2] / ratio[2]
            ego2egofeat[0,3] = - self.grid_config['x_bound'][0] / self.grid_config['x_bound'][2] / ratio[0]
            ego2egofeat[1,3] = - self.grid_config['y_bound'][0] / self.grid_config['y_bound'][2] / ratio[1]
            ego2egofeat[2,3] = - self.grid_config['z_bound'][0] / self.grid_config['z_bound'][2] / ratio[2]
            ego2egofeat = repeat(ego2egofeat, 'p q -> bs p q', bs=B).to(occ_feature)

            sampled_points = self.get_sampling_points().to(occ_feature.device) #(ds, hs, ws, 4)
            sampled_points = repeat(sampled_points, 'ds hs ws d -> b ds hs ws d', b=B)
            ego_points = self.back_projection(sampled_points, K, T)
            egofeat_points = rearrange(ego2egofeat, 'bs p q -> bs 1 1 1 p q') @ ego_points
            egofeat_points = egofeat_points.squeeze(-1)[...,:3] #(b, ds, hs, ws, 3)-(xyz)

        # import cv2
        # bev_feat = np.zeros((H_occ, W_occ), dtype=np.uint8)
        # egofeat_points_np = egofeat_points[0,:,:,:,:2].reshape(-1, 2).detach().cpu().numpy().astype(np.long)
        # for egofeat_point in egofeat_points_np:
        #     x, y = egofeat_point
        #     cv2.circle(bev_feat, (x, y), 1, (255, 255, 255), -1)
        # cv2.imwrite('proj_egopoint.png', bev_feat)

        nomralized_factor = torch.tensor([W_occ-1, H_occ-1, D_occ-1]).to(occ_feature.device)
        egofeat_points_norm = egofeat_points / nomralized_factor.view(1, 1, 1, 1, 3) * 2 - 1

        sampled_features = F.grid_sample(occ_feature, egofeat_points_norm, mode='bilinear', padding_mode='zeros', align_corners=True) #(b, C, ds, hs, ws)
        if self.with_DSE:
            sampled_features = self.DSE(sampled_features) + sampled_features
        sampled_features = torch.sum(sampled_features, dim=2)
        return sampled_features