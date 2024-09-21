import torch
import numpy as np
import torch.nn as nn
import pickle

import spconv.pytorch as spconv
from cldm.spconv_blocks import post_act_block, SparseBasicBlock

def get_sets_dict(filename):
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        return trajectories
        
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

class BoundingBoxEncoder(nn.Module):
    def __init__(self, out_channels, multires):
        super(BoundingBoxEncoder, self).__init__()
        embed_kwargs = {
                'include_input' : True,
                'input_dims' : 24,
                'max_freq_log2' : multires - 1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
        }
        self.embedder_obj = Embedder(**embed_kwargs)
        self.embedder = lambda x, eo=self.embedder_obj : eo.embed(x)
    
        self.mlp_for_geo = MLP(input_size=self.embedder_obj.out_dim, hidden_sizes=[512], output_size=out_channels)
        self.out_mlp = MLP(input_size=(out_channels+out_channels//2), hidden_sizes=[out_channels], output_size=out_channels)
        
    def forward(self, x):
        '''
        x is the bounding box infomation of a single image
        x: (B, M, (C + 24))
        B: batch size; M: maximum boxes of an image; C: clip or one-hot encoding of the box's label 
        '''
        B, M, D = x.shape[0], x.shape[1], x.shape[2]
        sem_inp, geo_inp = x[:,:,:D - 24], x[:,:,-24:]
        geo_with_pos = self.embedder(geo_inp) # add fourier position embedding
        geo_feats = self.mlp_for_geo(geo_with_pos)
        box_feats = torch.cat([geo_feats, sem_inp], axis=-1)
        out_feats = self.out_mlp(box_feats)
        return out_feats
                
class OccupancyEncoder(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, sparse_shape_xyz, **kwargs):
        super(OccupancyEncoder, self).__init__()
        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz
        self.with_fourier_emb = False
        if self.with_fourier_emb:
            multires = 7
            embed_kwargs = {
                'include_input' : True,
                'input_dims' : 1,
                'max_freq_log2' : multires - 1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
            }
            self.embedder_obj = Embedder(**embed_kwargs)
            self.embedder = lambda x, eo=self.embedder_obj : eo.embed(x)
            input_channel += self.embedder_obj.out_dim - 1
            
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

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

        # self.conv3 = spconv.SparseSequential(
        #     block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
        #     SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
        # )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*4, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True))
        
    def forward(self, voxel_features, coors):
        batch_size, N, D = voxel_features.shape[0], voxel_features.shape[1], voxel_features.shape[2]
        if self.with_fourier_emb:
            index = voxel_features[:,:,4:5] #(B, N, 1)
            index_embbed = self.embedder(index) #(B, N, F)
            voxel_features = torch.cat([voxel_features[:,:,0:4], index_embbed, voxel_features[:,:,5:]], -1)
            D = D + self.embedder_obj.out_dim - 1
        
        voxel_features = voxel_features.view(-1, D)
        coors = coors.view(-1, 3)[:, [2, 1, 0]] # change hwd-xyz to zyx-dwh
        batch_tensor = torch.arange(batch_size).unsqueeze(-1).repeat(1, N).view(-1).unsqueeze(-1).to(voxel_features.device)
        coors = torch.cat([batch_tensor, coors], axis=1)
        
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        
        x = self.conv_out(x_conv2)
        x = x.dense().permute(0,1,2,4,3)
        x = torch.rot90(x, k=1, dims=(3, 4))
        return {'x': x,  # BCDHW
                'pts_feats': [x]}
 
if __name__ == '__main__':
    BBEncoder = BoundingBoxEncoder(out_channels=1024, multires=3)
    sample_data_path = '/SSD_DISK/users/huangze/NuscenesControl/boundingbox/127.pkl'
    data = get_sets_dict(sample_data_path)
    
    s = np.random.rand(4, 100, 1024 + 24)
    s = torch.from_numpy(s).to(torch.float32)
    out = BBEncoder(s)