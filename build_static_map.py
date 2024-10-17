import os
import numpy as np
import math
from tqdm import tqdm
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, utils
import open3d as o3d
from pyntcloud import PyntCloud
import pandas as pd
import argparse

# Dictionary to map laser names to their string representations
laser_names_dict = {
    dataset_pb2.LaserName.TOP: 'TOP',
    dataset_pb2.LaserName.FRONT: 'FRONT',
    dataset_pb2.LaserName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.LaserName.SIDE_RIGHT: 'SIDE_RIGHT',
    dataset_pb2.LaserName.REAR: 'REAR',
}

def process_static_map(seq_path, seq_save_dir, voxel_size=0.05):
    """
    Process the static map from the Waymo dataset sequence file.

    Args:
        seq_path (str): Path to the sequence file.
        seq_save_dir (str): Directory to save the processed static map.
        voxel_size (float): Voxel size for downsampling.
    """
    print("Saving static map ...")
    datafile = WaymoDataFileReader(seq_path)
    output_static_map_path = f'{seq_save_dir}/downsampled{voxel_size}_static_map.ply'
    
    # Check if the static map already exists
    if os.path.exists(output_static_map_path):
        print("Static map already exists, skip...")
        return

    aligned_pts_3d = []
    lidar_positions_list = []

    # Iterate through each frame in the sequence
    for frame_id, frame in tqdm(enumerate(datafile)):
        pose = np.array(frame.pose.transform).reshape(4, 4)
        frame_pts_3d = []
        frame_lidar_positions = []

        # Process each laser in the frame
        for laser_name in laser_names_dict:
            laser = utils.get(frame.lasers, laser_name)
            laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
            extrinsic = np.array(laser_calibration.extrinsic.transform).reshape(4, 4)
            lidar_positions_veh_frame = np.dot(np.array([0, 0, 0, 1]), extrinsic.T)
            lidar_positions_world_frame = np.dot(pose, lidar_positions_veh_frame)
            ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)
            pcl, _ = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
            frame_pts_3d.append(pcl[:, :3])
            repeated_lidar_positions = np.repeat(lidar_positions_world_frame[np.newaxis, :3], pcl.shape[0], axis=0)
            frame_lidar_positions.append(repeated_lidar_positions)

        # Concatenate point clouds and lidar positions from all lasers
        pts_3d_concat = np.concatenate(frame_pts_3d, axis=0)
        lidar_positions_concat = np.concatenate(frame_lidar_positions, axis=0)

        # Filter out dynamic objects based on their speed
        for label in frame.laser_labels:
            box = label.box
            meta = label.metadata
            speed = np.linalg.norm([meta.speed_x, meta.speed_y])
            if speed < 1.:
                continue

            length, width, height = box.length, box.width, box.height
            tx, ty, tz = box.center_x, box.center_y, box.center_z
            heading = box.heading
            c, s = math.cos(heading), math.sin(heading)
            rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            obj_pose_vehicle = np.eye(4)
            obj_pose_vehicle[:3, :3] = rotz_matrix
            obj_pose_vehicle[:3, 3] = [tx, ty, tz]

            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            obj_pose_vehicle_inv = np.linalg.inv(obj_pose_vehicle)

            pts_3d_concat_hom = np.hstack((pts_3d_concat, np.ones((pts_3d_concat.shape[0], 1))))
            pts_3d_obj_vehicle = np.dot(obj_pose_vehicle_inv, pts_3d_concat_hom.T).T[:, :3]

            inside_bbox_mask = np.all((pts_3d_obj_vehicle >= bbox[0]) & (pts_3d_obj_vehicle <= bbox[1]), axis=1)
            pts_3d_concat = pts_3d_concat[~inside_bbox_mask]
            lidar_positions_concat = lidar_positions_concat[~inside_bbox_mask]

        # Transform points to world frame
        pts_3d_veh_frame_homo = np.hstack((pts_3d_concat, np.ones((pts_3d_concat.shape[0], 1))))
        pts_3d_world_frame_homo = np.dot(pts_3d_veh_frame_homo, pose.T)
        aligned_pts_3d.append(pts_3d_world_frame_homo[:, :3])
        lidar_positions_list.append(lidar_positions_concat)

    # Combine all frames into a single point cloud
    aligned_pts = np.vstack(aligned_pts_3d)
    lidar_positions = np.vstack(lidar_positions_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(aligned_pts)

    # Downsample the point cloud
    downsampled_pcd, _, point_indices = pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size, 
        min_bound=pcd.get_min_bound(), 
        max_bound=pcd.get_max_bound()
    )

    # Get the first indices of the downsampled points
    first_indices = np.array([indices[0] for indices in point_indices])
    lidar_pos_downsampled = lidar_positions[first_indices]
    downsampled_pts = np.asarray(downsampled_pcd.points)

    # Create a DataFrame for the downsampled points and their sensor positions
    pdata = pd.DataFrame({
        'x': downsampled_pts[:, 0],
        'y': downsampled_pts[:, 1],
        'z': downsampled_pts[:, 2],
        'sensor_x': lidar_pos_downsampled[:, 0],
        'sensor_y': lidar_pos_downsampled[:, 1],
        'sensor_z': lidar_pos_downsampled[:, 2]
    })

    # Save the downsampled point cloud to a file
    pcloud = PyntCloud(pdata)
    pcloud.to_file(output_static_map_path)
    print(f"Static map successfully aligned and saved to: {output_static_map_path}")

def main():
    """
    Main function to parse arguments and process the static map.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_path', type=str, required=True, help='Path to the sequence file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the processed static map')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size for downsampling')
    args = parser.parse_args()

    process_static_map(args.seq_path, args.save_dir, args.voxel_size)

if __name__ == '__main__':
    main()