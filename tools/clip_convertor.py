from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
import torch
import numpy as np
import os

model = FrozenOpenCLIPEmbedder(arch="ViT-H-14", version="laion2b_s32b_b79k", device="cpu", max_length=1,
                freeze=True, layer="last")

txt_list_occupancy = ["barrier", "bicycle", "bus", "car", "construction vehicle", "motocycle", "pedestrian",
            "traffic cone", "trailer", "truck", "driveable surface", "other flat", "sidewalk",
            "terrain", "construction buildings", "vegetation"]
txt_list_hdmap = ['road_segment', 'lane', 'ped crossing', 'walkway', 'stop line', 'carpark area', 'border']

cross_domain_list = [
        "purple bus",
        "purple car",
        "purple construction vehicle",
        "purple traffic cone",
        "purple trailer",
        "purple truck",
        "purple vegetation",
]
# for occupancy
out_path = './output/clip_openclip/'
for text in txt_list_occupancy:
    with torch.no_grad():
        out = model.encode(text)
        out_name = text.replace(' ', '_')
        np.save(os.path.join(out_path, out_name + '.npy'), out.squeeze().cpu().numpy()) 

# for hdmap
out_path = './output/clip_openclip/hdmap/'
for text in txt_list_hdmap:
    with torch.no_grad():
        out = model.encode(text)
        out_name = text.replace(' ', '_')
        np.save(os.path.join(out_path, out_name + '.npy'), out.squeeze().cpu().numpy()) 