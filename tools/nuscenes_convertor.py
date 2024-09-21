import numpy as np
import descartes
import shutil
import argparse
import os
import json
import matplotlib.pyplot as plt
import pdb
import pickle
import torch
import torchvision

from PIL import Image
from tqdm import tqdm
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from torch import nn
from diffusers import AutoencoderKL

from nuscenes.map_expansion.arcline_path_utils import discretize_lane, ArcLinePath
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility, transform_matrix
from nuscenes.utils.color_map import get_colormap
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

from tools.nusenes_occ_dataset import NuScenes_Occ # the expanded occupancy dataset

parser = argparse.ArgumentParser(description='The script for generating source/target and conditions from Nuscenes for Controlnet')
parser.add_argument('--nusc_root', type=str)
parser.add_argument('--nusc_occ_root', type=str)
parser.add_argument('--out_root', type=str)
parser.add_argument('--conditions', default={'source', 'target', 'text', 'hdmap', 'occupancy', 'boundingbox', 'pose'}, type=dict)
parser.add_argument('--cover', help='re-generate extisting data', action="store_true")
parser.add_argument('--vae', help='start up mae', action="store_true")
args = parser.parse_args()

nusc_map_dict = {
    'singapore-onenorth': NuScenesMap(dataroot=args.nusc_root, map_name='singapore-onenorth'),
    'boston-seaport': NuScenesMap(dataroot=args.nusc_root, map_name='boston-seaport'),
    'singapore-hollandvillage': NuScenesMap(dataroot=args.nusc_root, map_name='singapore-hollandvillage'),
    'singapore-queenstown': NuScenesMap(dataroot=args.nusc_root, map_name='singapore-queenstown'),
}


whole_layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area'] # for hdmap old version
sensors = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'] # surrround cameras

def encode_img(input_img):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img * 2 - 1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def decode_img(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach()
    return image

def init_dirs(root, conditions):
    if not os.path.exists(root):
        os.makedirs(root)
    
    for folder_name in conditions:
        folder_path = os.path.join(root, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def scene_level_prompt_generator(scene_description, place=None):
    scene_description = scene_description.lower()
    scene_descs = scene_description.split(', ')
    
    basic_prompt = 'Realistic autonousmous driving scene'
    
    # add basic description
    with_basic_desc = False
    basic_descs = ['night', 'rain', 'after rain']
    for basic_desc in basic_descs:
        if basic_desc in scene_descs:
            with_basic_desc = True 
            if basic_desc == 'night':
                basic_prompt += ' at night'
            elif basic_desc == 'rain':
                basic_prompt += ' in rain'
            elif basic_desc == 'after rain':
                basic_prompt += ' after rain'
            scene_descs.remove(basic_desc)
    if not with_basic_desc:
        basic_prompt += ' at day'
        
    # add speed description
    with_speed_desc = False
    speed_descs = ['high speed']
    for speed_desc in speed_descs:
        if speed_desc in scene_descs:
            basic_prompt = basic_prompt + ' with ' + speed_desc
            scene_descs.remove(speed_desc)
            with_speed_desc = True
    
    
    
    # add location description
    if place is not None:
        basic_prompt = basic_prompt + ' in ' + place
        
    basic_prompt += '.'
    
    # add driving description
    with_drive_desc = False
    driving_prompt = ' The ego-vehicle is'
    drive_descs = ['wait', 'follow', 'overtake', 'arrive', 'turn', 'start', 'move', 'drive', 'cross ', 'exit', 'pass', 'attempt', 'reduce', 'go', 'stop ', 'slow']
    for scene_desc in scene_descs:
        for drive_desc in drive_descs:
            if scene_desc.startswith(drive_desc):
                if not with_drive_desc:
                    driving_prompt = driving_prompt + ' ' + scene_desc
                    with_drive_desc = True
                else:
                    driving_prompt = driving_prompt + ' and then ' + scene_desc
                scene_descs.remove(scene_desc)
    if not with_drive_desc:
        driving_prompt = ''
    else:
        driving_prompt += '.'
        
    # add environmental description
    with_envir_desc = False
    environment_prompt = ' The driving scene is in'
    envir_descs = ['big street', 'narrow street', 'dead end street', 'busy street', 'long street', 'very dense traffic', 'low traffic', 'no traffic', 'heavy traffic', 'traffic chaos', 'dense traffic', 'nature']
    for scene_desc in scene_descs:
        for envir_desc in envir_descs:
            if scene_desc.startswith(envir_desc):
                if not with_drive_desc:
                    environment_prompt = environment_prompt + ' ' + scene_desc
                    with_envir_desc = True
                else:
                    environment_prompt = environment_prompt + ' and ' + scene_desc
                scene_descs.remove(scene_desc)
    if not with_envir_desc:
        environment_prompt = ' The driving scene is in urban'
        
    # add object description
    with_object_desc = False
    object_prompt = ', with '
    for scene_desc in scene_descs:
        if not with_object_desc:
            object_prompt = object_prompt + scene_desc
            with_object_desc = True
        else:
            object_prompt = object_prompt + ' and ' + scene_desc
    if not with_object_desc:
       object_prompt = '.'
    else:
       object_prompt += '.' 
    return basic_prompt + driving_prompt + environment_prompt + object_prompt
    
def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range])

    ax.set_xlim([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim([y_middle - plot_radius, y_middle + plot_radius])
    
def render_map_in_image(nusc_map: NuScenesMap,
                            nusc: NuScenes,
                            sample_token: str,
                            camera_channel: str = 'CAM_FRONT',
                            alpha: float = 0.3,
                            patch_radius: float = 40,
                            min_polygon_area: float = 10,
                            render_behind_cam: bool = True,
                            render_outside_im: bool = True,
                            layer_names: List[str] = None,
                            verbose: bool = True,
                            out_path: str = None,
                            fig = None
                            ):
        near_plane = 1e-8

        if verbose:
            print('Warning: Note that the projections are not always accurate as the localization is in 2d.')

        # Default layers.
        if layer_names is None:
            layer_names = whole_layer_names

        # Check layers whether we can render them.
        for layer_name in layer_names:
            assert layer_name in nusc_map.explorer.map_api.non_geometric_polygon_layers, \
                'Error: Can only render non-geometry polygons: %s' % layer_names

        # Check that NuScenesMap was loaded for the correct location.
        sample_record = nusc.get('sample', sample_token)
        scene_record = nusc.get('scene', sample_record['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']
        assert nusc_map.explorer.map_api.map_name == log_location, \
            'Error: NuScenesMap loaded for location %s, should be %s!' % (nusc_map.explorer.map_api.map_name, log_location)

        # Grab the front camera image and intrinsics.
        cam_token = sample_record['data'][camera_channel]
        cam_record = nusc.get('sample_data', cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)
        im = Image.open(cam_path)
        im_size = im.size
        cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

        # Retrieve the current map.
        poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        box_coords = (
            ego_pose[0] - patch_radius,
            ego_pose[1] - patch_radius,
            ego_pose[0] + patch_radius,
            ego_pose[1] + patch_radius,
        )
        records_in_patch = nusc_map.explorer.get_records_in_patch(box_coords, layer_names, 'intersect')

        # Init axes.
        #fig = plt.figure(figsize=(9, 16))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-40., 40.)
        ax.set_ylim(-40., 40.)
        #ax.imshow(np.zeros_like(im))
        hdmap_list = []

        # Retrieve and render each record.
        for layer_name in layer_names:
            for token in records_in_patch[layer_name]:
                record = nusc_map.explorer.map_api.get(layer_name, token)
                if layer_name == 'drivable_area':
                    polygon_tokens = record['polygon_tokens']
                else:
                    polygon_tokens = [record['polygon_token']]

                for polygon_token in polygon_tokens:
                    polygon = nusc_map.explorer.map_api.extract_polygon(polygon_token)

                    # Convert polygon nodes to pointcloud with 0 height.
                    points = np.array(polygon.exterior.xy)
                    points = np.vstack((points, np.zeros((1, points.shape[1]))))

                    # Transform into the ego vehicle frame for the timestamp of the image.
                    points = points - np.array(poserecord['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points) #(3, N)
                    
                    # take out the points in the ego-vehicle system
                    points = points[:2, :]
                    points_poly = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                    polygon_proj = Polygon(points_poly)

                    # Filter small polygons
                    if polygon_proj.area < min_polygon_area:
                        continue
                    
                    label = layer_name
                    
                    #ax.scatter(points[0,:], points[1,:], c=nusc_map.explorer.color_map[layer_name], cmap='viridis', s=1)
                    for n in range(points[0].shape[0]):
                        if n == points[0].shape[0] - 1:
                            p = 0
                        else:
                            p = n + 1
                        plt.plot([points[0][n], points[0][p]], [points[1][n], points[1][p]], linewidth=7, markersize=1, color='black')
                    ax.add_patch(descartes.PolygonPatch(polygon_proj, fc=nusc_map.explorer.color_map[layer_name], alpha=alpha, label=label))
                    
        
        # draw ego-car
        #plt.plot([0., 40.], [0., 0.])
        
        # Display the image.
        plt.axis('off')
        #plt.minorticks_on()
        #plt.xlabel('x-axis')
        #plt.ylabel('y-axis')
        set_axes_equal(ax)

        if out_path is not None:
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.clf()

def localization_description(front_corners, front_depth, rear_corners, rear_depth, H=900, W=1600, d_far=50., d_near=5.):
    front_center = front_corners.mean(axis=0)
    rear_center = rear_corners.mean(axis=0)
    front_depth_center = front_depth.mean()
    rear_depth_center = rear_depth.mean()
    
    basic_prompt = 'is '
    
    # close or far way in camera
    if front_depth_center > d_far or rear_depth_center > d_far:
        # too far
        return None
    elif front_depth_center < d_near or rear_depth_center < d_near:
        basic_prompt += 'very close '
    else:
        depth = 'normal'
    
    # left right or middle in camera
    if front_center[0] < W/3:
        basic_prompt += 'on the left side,'
    elif front_center[0] > 2*W/3:
        basic_prompt += 'on the right side,'
    else:
        basic_prompt += 'in the middle,'
    
    # forwarding or not
    if front_depth_center > rear_depth_center:
        basic_prompt += ' back to camera'
    else:
        basic_prompt += ' face to camera'
        
    # partly outside or not
    if front_center[0] > W or front_center[0] < 0 or rear_center[0] > W or rear_center[0] < 0 or front_center[1] > H or front_center[1] < 0 or rear_center[1] > H or rear_center[1] < 0:
        basic_prompt += ', partly outside the image'
        
    return basic_prompt
    
def object_description(name, current_attr):
    object_name = name[-1].replace('_', ' ')
    if object_name == 'rigid':
        object_name = 'rigid bus'
    elif object_name == 'bendy':
        object_name = 'blendy bus'
    
    if current_attr is None:
        state = ''
    else:
        state = current_attr[-1].replace('_', ' ')
    if name[-1] == 'bicycle':
        if state != '':
            object = 'a ' + object_name + ' ' + state
        else:
            object = 'a ' + object_name + ' '
    else:
        if state != '':
            object = 'a ' + state + ' ' + object_name
        else:
            object = 'a ' + object_name
    return object


if __name__ == "__main__":
    # load origin nuscenes and nuscenes occupancy
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.nusc_root, verbose=True)
    nusc_occ = NuScenes_Occ(dataroot=args.nusc_occ_root)
    nusc_occ_data = nusc_occ.get_data_dict()
    
    # prepare files for output
    init_dirs(args.out_root, args.conditions)
    data_meta_list = []
    fig = plt.figure(figsize=(24, 24))
    
    # load vae model
    if args.vae:
        vae = AutoencoderKL.from_pretrained("./models/vae", use_safetensors=False).cuda()
    
    # prepare hyperparameters
    D = 21 # occupancy dimensions
    
    # start loop
    for scene_index, single_scene in enumerate(tqdm(nusc.scene)):
                
        # take out all tokens of a single scene
        sample_token_list = []
        sample_token = single_scene['first_sample_token']
        my_sample = nusc.get('sample', sample_token)
        sample_token_list.append(sample_token)
        
        scene_token = single_scene['token']
        scene_record = nusc.get('scene', scene_token)
        log_record = nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']
        nusc_map = nusc_map_dict[log_location]
        
        # generate global text prompt for one scene
        if 'text' in args.conditions:
            text_prompt = scene_level_prompt_generator(single_scene['description'], place=log_location.replace('-', ' '))
            print(text_prompt)
        
        while my_sample['next'] != '':
            sample_token = my_sample['next']
            my_sample = nusc.get('sample', sample_token)
            sample_token_list.append(sample_token)
            
        # for each sample token in scene
        for i, sample_token in enumerate(sample_token_list):
            my_sample = nusc.get('sample', sample_token)
            
            # occupancy serves as the basic feature for generation
            rendered_occupancy_save_path = args.out_root + 'occupancy/' + f'{scene_index}_{i}' + '.pth'
            occupancy_path = nusc_occ_data[single_scene['name']][sample_token]
            
            if not os.path.exists(rendered_occupancy_save_path) or args.cover:
                # load occpuancy around ego-vehicle
                semantics, mask_lidar, mask_camera = nusc_occ.load_occupancy(occupancy_path)
                occ_grid_all = torch.zeros(semantics.shape).cuda().unsqueeze(-1).repeat(1, 1, 1, D) # D = num_classes (17) + latent dims (4)
                semantic_tensor = torch.from_numpy(semantics)
                
            # render hdmap elements in occupancy
            hd_out_path = args.out_root + 'hdmap_v2/' + str(scene_index) + '_' + str(i) + '.jpg'
            if 'hdmap' in args.conditions and not os.path.exists(hd_out_path):
                render_map_in_image(nusc_map, nusc, sample_token, camera_channel='CAM_FRONT', alpha=1.0, verbose=False, out_path=hd_out_path, fig=fig)
                
            # for each camera, bestow occupancy with image reconstruction feature
            for cam_name in sensors:
                camera_data = nusc.get('sample_data', my_sample['data'][cam_name]) #['token', 'timestamp', 'prev', 'next', 'scene_token', 'data', 'anns']
                camera_ego_pose = nusc.get('ego_pose', camera_data['ego_pose_token']) # global pose
                camera_clb_pose = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token']) # calibration pose
                camera_anns = my_sample['anns']
                cur_im_path, _, _ = nusc.get_sample_data(camera_data['token']) # camera image path
                
                
                if not os.path.exists(rendered_occupancy_save_path) or args.cover:
                    cur_im = torch.tensor(np.array(Image.open(cur_im_path)).astype(np.float32) / 256).permute(2, 0, 1)[None].cuda()
                    latents = encode_img(cur_im)
                    # bestow occupancy with image reconstruction feature
                    occ_grid_all = nusc_occ.render_occupancy_appearance(semantic_tensor, camera_clb_pose=camera_clb_pose, feature_map=latents, occ_grid_all=occ_grid_all)
                    

                data_meta_list.append({
                    "global_prompt": text_prompt,
                    "hdmap": hd_out_path,
                    "pose_token": camera_data['calibrated_sensor_token'],
                    "occupancy": rendered_occupancy_save_path,
                    "image_token": camera_data['token'],
                })
            
            #save the world feature   
            if not os.path.exists(rendered_occupancy_save_path) or args.cover:
                torch.save(occ_grid_all.cpu(), rendered_occupancy_save_path)
                
            
    with open(args.out_root + "prompt_v3.json", 'w') as out_file:
        json.dump(data_meta_list, out_file, indent=4)      
        