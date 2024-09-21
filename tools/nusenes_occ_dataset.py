import os
import numpy as np
import open3d as o3d
import pickle
import torch
import math
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.ndimage import zoom
from pyquaternion import Quaternion

from utils.pointcloud import quaternion_to_rotation, get_transform_from_rotation_translation, apply_transform, inverse_transform
from scipy.spatial.transform import Rotation as R

LINE_SEGMENTS = [
    [4, 0], [3, 7], [5, 1], [6, 2],  # lines along x-axis
    [5, 4], [5, 6], [6, 7], [7, 4],  # lines along x-axis
    [0, 1], [1, 2], [2, 3], [3, 0]]  # lines along y-axis
colors_map_wrong = np.array(
    [
        # [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
    ])
colors_map = np.array(
    [
        [0,   0,   0, 255],    # 0 undefined
        [112, 128, 144, 255],  # 1 barrier  orange
        [220, 20, 60, 255],    # 2 bicycle  Blue
        [255, 127, 80, 255],   # 3 bus  Darkslategrey
        [255, 158, 0, 255],    # 4 car  Crimson
        [233, 150, 70, 255],   # 5 construction vehicle  Orangered
        [255, 61, 99, 255],    # 6 motocycle  Darkorange
        [0, 0, 230, 255],      # 7 pedestrian  Darksalmon
        [47, 79, 79, 255],     # 8 traffic_cone  Red
        [255, 140, 0, 255],    # 9 trailer
        [255, 99, 71, 255],    # 10 truck  Slategrey
        [0, 207, 191, 255],    # 11 driveable_surface
        [175, 0, 75, 255],     # 12 other_flat
        [75, 0, 75, 255],      # 13 sidewalk
        [112, 180, 60, 255],   # 14 terrain
        [222, 184, 135, 255],  # 15 manmade
        [0, 175, 0, 255],      # 16 vegetation
        #[255, 255, 255, 255], # 17 free
    ])
color = colors_map[:, :3] / 255

def voxel2points(voxel, voxelSize, range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], ignore_labels=[17, 255], with_idx=False):
    if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
    mask = torch.zeros_like(voxel, dtype=torch.bool)
    for ignore_label in ignore_labels:
        mask = torch.logical_or(voxel == ignore_label, mask)
    mask = torch.logical_not(mask)
    occIdx = torch.where(mask)
    # points = torch.concatenate((np.expand_dims(occIdx[0], axis=1) * voxelSize[0], \
    #                          np.expand_dims(occIdx[1], axis=1) * voxelSize[1], \
    #                          np.expand_dims(occIdx[2], axis=1) * voxelSize[2]), axis=1)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + voxelSize[0] / 2 + range[0], \
                        occIdx[1][:, None] * voxelSize[1] + voxelSize[1] / 2 + range[1], \
                        occIdx[2][:, None] * voxelSize[2] + voxelSize[2] / 2 + range[2]), dim=1)
    if with_idx:
        return points, voxel[occIdx], occIdx
    else:
        return points, voxel[occIdx]

def voxel_profile(voxel, voxel_size):
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)

def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    #R = rotz(1 * heading_angle)
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    #corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d

def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    ego_voxel_num = ego_xdim * ego_ydim * ego_zdim
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz

def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, obj_bboxes=None, voxelize=False, bbox_corners=None, linesets=None, ego_pcd=None, scene_idx=0, frame_idx=0, large_voxel=True, voxel_size=0.4, with_mesh=False, view_pose=None, intrinsics=None, H=900, W=1600, save=False, save_path=None) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(str(scene_idx), width=W, height=H, visible=(not save))

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    
    if with_mesh:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.6, origin=[0, 0, 0])

    pcd.points = o3d.utility.Vector3dVector(points)
    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)
    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3)))
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
    
    if with_mesh:
        vis.add_geometry(mesh_frame)
        
    vis.add_geometry(pcd)    
    if voxelize:
        vis.add_geometry(line_sets)
        
    view_control = vis.get_view_control()
    if intrinsics is not None and view_pose is not None:
        fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
        init_param = view_control.convert_to_pinhole_camera_parameters()
        init_param.intrinsic.width = W
        init_param.intrinsic.height = H
        init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
        init_param.extrinsic = inverse_transform(view_pose)
        view_control.convert_from_pinhole_camera_parameters(init_param, True)
    else:  
        view_control.set_lookat(np.array([0, 0, 0]))
    vis.poll_events()
    vis.update_renderer()
    
    if save:
        vis.capture_screen_image(save_path, do_render=True)
    return vis
    
def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class NuScenes_Occ(object):
    def __init__(self, dataroot, clip_path='./output/clip/'):
        self.dataroot = dataroot
        self.clip_path = clip_path
        self.data_dict = {}
        
        subdirectories = [d for d in os.listdir(self.dataroot) if os.path.isdir(os.path.join(self.dataroot, d))]
        for subdir in subdirectories:
            self.data_dict[subdir] = {}
            subdir_path = os.path.join(self.dataroot, subdir)
            subtokens = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
            for subtoken in subtokens:
                self.data_dict[subdir][subtoken] = os.path.join(self.dataroot, subdir, subtoken, 'labels.npz')
                
        self.voxelSize = [0.4, 0.4, 0.4]
        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.ignore_labels = [17, 255]
        self.vis_voxel_size = 0.4
        
        self.occupancy_labels = ["barrier", "bicycle", "bus", "car", "construction vehicle", "motocycle", "pedestrian",
                                 "traffic cone", "trailer", "truck", "driveable surface", "other flat", "sidewalk",
                                 "terrain", "manmade", "vegetation"]
        
        #self.load_clip_feature()
    
    def get_data_dict(self):
        return self.data_dict
    
    def load_clip_feature(self):
        clip_features = []
        for label in self.occupancy_labels:
            name_for_clip = label.replace(' ', '_')
            text_feature = np.load(self.clip_path + name_for_clip + '.npy')
            clip_features.append(text_feature)
        
        zero_feature, one_feature = np.zeros_like(clip_features[0]) , np.ones_like(clip_features[0])
        clip_features.append(zero_feature)
        clip_features.insert(0, one_feature)
        self.clip_features = clip_features
        
    def load_occupancy(self, path):
        occupancy = np.load(path) #['semantics', 'mask_lidar', 'mask_camera']
        semantics, mask_lidar, mask_camera = occupancy['semantics'], occupancy['mask_lidar'], occupancy['mask_camera'] # all shaped as (200, 200, 16)
        return semantics, mask_lidar, mask_camera
        
    def render_occupancy_appearance(self, semantics, mask_lidar=None, mask_camera=None, camera_clb_pose=None, save_path=None,
                         feature_map=None, occ_grid_all=None):
        voxels = semantics
        if mask_lidar is not None:
            semantics[mask_lidar == 1] = 17
        elif mask_camera is not None:
            semantics[mask_camera == 1] = 17
            
        if camera_clb_pose is not None:
            clb_R, clb_t = Quaternion(camera_clb_pose['rotation']).rotation_matrix, camera_clb_pose['translation']
            clb_T = get_transform_from_rotation_translation(clb_R, clb_t) # cam2ego
            cmr_K = camera_clb_pose['camera_intrinsic']
        else:
            clb_T, cmr_K = None, None     
        
        points, labels, occIdx = voxel2points(voxels, self.voxelSize, range=self.point_cloud_range, ignore_labels=self.ignore_labels, with_idx=True)
        bboxes = voxel_profile(points, self.voxelSize)
        bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
        
        ###################################################
        H, W = 900, 1600
        points_all = torch.cat([points.unsqueeze(1), bboxes_corners], dim=1).view(-1, 3)
        points_all = apply_transform(points_all, torch.from_numpy(inverse_transform(clb_T)).float()).cuda()
        points_proj = (torch.matmul(torch.tensor(cmr_K).float().cuda(), points_all.T)).T
        points_x = (points_proj[:, 0] / points_proj[:, 2]).cpu().numpy()
        points_y = (points_proj[:, 1] / points_proj[:, 2]).cpu().numpy()
        points_z = (points_proj[:, 2]).cpu().numpy()
        points_xy = points_proj[:, 0:2] / points_proj[:, 2:3]
        
        indices_x = np.where((0<=points_x)&(points_x<=W))[0]
        indices_y = np.where((0<=points_y)&(points_y<=H))[0]
        indices_z = np.where((0<=points_z))[0]
        xy, _, _ = np.intersect1d(indices_x, indices_y, return_indices=True)
        visible_indices, _, _ = np.intersect1d(xy, indices_z, return_indices=True)
        visible_indices = torch.from_numpy(visible_indices).cuda()
        
        points_xy_norm = points_xy / torch.tensor([[W, H]], device=points_xy.device) * 2 - 1
        points_feats = torch.nn.functional.grid_sample(feature_map, points_xy_norm[None,None].float())[0,:,0].T
        points_feats_all = torch.zeros_like(points_feats)
        points_feats_all[visible_indices] = points_feats[visible_indices]
        points_feats_all = points_feats_all.view(-1,9,points_feats_all.shape[-1])
        points_feats_all = (points_feats_all * torch.tensor([0.5, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625], device=points_feats_all.device)[None,:,None]).sum(1)
        labels_onehot = torch.nn.functional.one_hot(labels.long(), num_classes=17).cuda()
        occ_feats_nonempty = torch.cat([points_feats_all, labels_onehot], dim=-1)
        
        voxel_visible_mask = torch.zeros(occIdx[0].shape[0] * 9, dtype=torch.bool)
        voxel_visible_mask[visible_indices] = True
        voxel_visible_mask = voxel_visible_mask.view(-1,9).any(-1)
        
        occ_grid_all[occIdx[0][voxel_visible_mask], occIdx[1][voxel_visible_mask], occIdx[2][voxel_visible_mask]] = occ_feats_nonempty[voxel_visible_mask]
        
        return occ_grid_all
    
    def render_occupancy(self, semantics, mask_lidar=None, mask_camera=None, camera_clb_pose=None, save=False, save_path=None):
        voxels = semantics
        if mask_lidar is not None:
            semantics[mask_lidar == 0] = 17
        elif mask_camera is not None:
            semantics[mask_camera == 0] = 17
            
        if camera_clb_pose is not None:
            q = camera_clb_pose['rotation']      
            clb_R, clb_t = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]).rotation_matrix, camera_clb_pose['translation']
            clb_T = get_transform_from_rotation_translation(clb_R, clb_t)
            cmr_K = camera_clb_pose['camera_intrinsic']
        else:
            clb_T, cmr_K = None, None     
        
        points, labels = voxel2points(voxels, self.voxelSize, range=self.point_cloud_range, ignore_labels=self.ignore_labels)
        points = points.numpy()
        labels = labels.numpy()
        pcd_colors = color[labels.astype(int) % len(color)]
        bboxes = voxel_profile(torch.tensor(points), self.voxelSize)
        ego_pcd = o3d.geometry.PointCloud()
        ego_points = generate_the_ego_car()
        ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
        bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
        bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
        edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
        edges = edges + bases_[:, None, None]
        vis = show_point_cloud(points=points, colors=True, points_colors=pcd_colors, voxelize=False, obj_bboxes=None,
                            bbox_corners=bboxes_corners.numpy(), linesets=edges.numpy(), ego_pcd=ego_pcd, large_voxel=True, voxel_size=self.vis_voxel_size, with_mesh=False,
                            view_pose=clb_T, intrinsics=cmr_K, save=save, save_path=save_path)
        if not save:
            vis.run()
        vis.destroy_window()
        del vis
        
    def generate_coarse_mask(self, x, y, c, t, zoom_factor=2.0):
        H, W = 900, 1600
        x = x[c == t].astype(int)
        y = y[c == t].astype(int)
        mask = np.zeros((H, W))
        
        mask[y, x] = 255.
        
        #mask = zoom(mask, zoom_factor, order=0)
        #mask = (mask >= 0.5).astype(np.uint8)
        return mask
        
    def render_occupancy_to_image(self, image_path, semantics, camera_clb_pose, mask_camera=None):
        q = camera_clb_pose['rotation']      
        clb_R, clb_t = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]).rotation_matrix, camera_clb_pose['translation']
        clb_T = get_transform_from_rotation_translation(clb_R, clb_t)
        cmr_K = camera_clb_pose['camera_intrinsic']
        
        if image_path is not None:
            image = Image.open(image_path)
            image_array = np.array(image)
            H, W = image_array.shape[0], image_array.shape[1]
        else:
            H, W = 900, 1600
            image_array = np.ones((H, W))
        
        if mask_camera is not None:
            semantics[mask_camera == 0] = 17
        voxels = semantics       
        points, labels = voxel2points(voxels, self.voxelSize, range=self.point_cloud_range, ignore_labels=self.ignore_labels)
        points = points.numpy()
        labels = labels.numpy()
        
        points = apply_transform(points, inverse_transform(clb_T))
        points_proj = (np.dot(cmr_K, points.T)).T
        points_x = points_proj[:, 0] / points_proj[:, 2]
        points_y = points_proj[:, 1] / points_proj[:, 2]
        points_z = points_proj[:, 2]
        
        indices_x = np.where((0<=points_x)&(points_x<=W))[0]
        indices_y = np.where((0<=points_y)&(points_y<=H))[0]
        indices_z = np.where((0<=points_z))[0]
        xy, _, _ = np.intersect1d(indices_x, indices_y, return_indices=True)
        visible_indices, _, _ = np.intersect1d(xy, indices_z, return_indices=True)
        
        visible_points_x = points_x[visible_indices]
        visible_points_y = points_y[visible_indices]
        visible_points_z = points_z[visible_indices]
        visible_labels = labels[visible_indices]
        pcd_colors = color[visible_labels.astype(int) % len(color)]
        
        plt.imshow(image_array)
        plt.scatter(visible_points_x, visible_points_y, marker='o', color=pcd_colors, s=20)
        plt.title('Image')    
        
        #mask = self.generate_coarse_mask(visible_points_x, visible_points_y, visible_labels, 4)
        #plt.subplot(1, 2, 2)
        #plt.imshow(mask, cmap='viridis')
        #plt.title('Mask')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
        
    def render_occupancy_mtplb(self, semantics, mask_lidar=None, mask_camera=None):
        voxels = semantics
        points, labels = voxel2points(voxels, self.voxelSize, range=self.point_cloud_range, ignore_labels=self.ignore_labels)
        points = points.numpy()
        labels = labels.numpy()
        pcd_colors = color[labels.astype(int) % len(color)]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=pcd_colors, cmap='viridis')
        ax.set_box_aspect([1.0, 1.0, 1.0])
        
        set_axes_equal(ax)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        plt.show()