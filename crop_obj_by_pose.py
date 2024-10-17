import open3d as o3d
import numpy as np
import mathutils
import argparse

def load_mesh(file_path):
    """
    Load the mesh from the specified file.
    
    Args:
        file_path (str): Path to the mesh file.
    
    Returns:
        mesh (open3d.geometry.TriangleMesh): Loaded mesh.
    """
    return o3d.io.read_triangle_mesh(file_path)

def load_camera_pose(file_path):
    """
    Load the camera pose from the specified file.
    
    Args:
        file_path (str): Path to the camera pose file.
    
    Returns:
        camera_pose (np.ndarray): Camera pose as a 4x4 matrix.
    """
    return np.loadtxt(file_path).reshape((4, 4))

def compute_relative_rotation(quaternion_src, quaternion_tgt):
    """
    Compute the relative rotation matrix between two quaternions.
    
    Args:
        quaternion_src (mathutils.Quaternion): Source quaternion.
        quaternion_tgt (mathutils.Quaternion): Target quaternion.
    
    Returns:
        relative_rotation_matrix (np.ndarray): Relative rotation matrix.
    """
    quaternion_src_inv = quaternion_src.inverted()
    relative_rotation = quaternion_tgt @ quaternion_src_inv
    return np.array(relative_rotation.to_matrix())

def create_oriented_bounding_box(center, rotation_matrix, extents):
    """
    Create an oriented bounding box.
    
    Args:
        center (np.ndarray): Center of the bounding box.
        rotation_matrix (np.ndarray): Rotation matrix of the bounding box.
        extents (np.ndarray): Extents of the bounding box.
    
    Returns:
        obb (open3d.geometry.OrientedBoundingBox): Oriented bounding box.
    """
    return o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extents)

def crop_mesh(mesh, obb):
    """
    Crop the mesh using the oriented bounding box.
    
    Args:
        mesh (open3d.geometry.TriangleMesh): Mesh to be cropped.
        obb (open3d.geometry.OrientedBoundingBox): Oriented bounding box.
    
    Returns:
        cropped_mesh (open3d.geometry.TriangleMesh): Cropped mesh.
    """
    return mesh.crop(obb)

def compute_inverse_transformation(matrix):
    """
    Compute the inverse transformation matrix.
    
    Args:
        matrix (np.ndarray): Transformation matrix.
    
    Returns:
        inverse_matrix (np.ndarray): Inverse transformation matrix.
    """
    R_matrix = matrix[:3, :3]
    t = matrix[:3, 3]
    R_inv = R_matrix.T
    t_inv = -R_inv @ t
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = R_inv
    inverse_matrix[:3, 3] = t_inv
    return inverse_matrix

def transform_vertices(vertices, transformation_matrix, rotation_matrix):
    """
    Transform the vertices using the transformation and rotation matrices.
    
    Args:
        vertices (np.ndarray): Vertices to be transformed.
        transformation_matrix (np.ndarray): Transformation matrix.
        rotation_matrix (np.ndarray): Rotation matrix.
    
    Returns:
        transformed_vertices (np.ndarray): Transformed vertices.
    """
    return ((transformation_matrix[:3, :3] @ vertices.T + transformation_matrix[:3, 3:4]).T) @ rotation_matrix.T

def save_mesh(file_path, mesh):
    """
    Save the mesh to the specified file.
    
    Args:
        file_path (str): Path to the output file.
        mesh (open3d.geometry.TriangleMesh): Mesh to be saved.
    """
    o3d.io.write_triangle_mesh(file_path, mesh)

def main():
    parser = argparse.ArgumentParser(description='Process some paths and parameters.')
    parser.add_argument('--mesh_file_path', type=str, default='waymo_mesh_save.obj', help='Path to the mesh file')
    parser.add_argument('--processed_waymo_dataset_path', type=str, default='/home/joey/dataset/waymo_for_s3gaussian/processed/training/', help='Path to the processed Waymo dataset')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    parser.add_argument('--seq', type=int, default=69, help='Sequence number')
    parser.add_argument('--frame', type=int, default=18, help='Frame number')

    args = parser.parse_args()

    mesh_file_path = args.mesh_file_path
    processed_waymo_dataset_path = args.processed_waymo_dataset_path
    output_dir = args.output_dir
    seq = args.seq
    frame = args.frame

    print(f'Processing mesh file {mesh_file_path} for sequence {seq} and frame {frame}')
    mesh = load_mesh(mesh_file_path)

    # Load the camera pose in the world frame
    camera_pos_world_frame_path = f'{processed_waymo_dataset_path}/{seq:03d}/ego_pose/{frame:06d}_0.txt'
    print(f'Loading camera pose from {camera_pos_world_frame_path}')
    camera_pose_world_frame = load_camera_pose(camera_pos_world_frame_path)
    print(camera_pose_world_frame)

    # Define the source and target quaternions
    quaternion_src = mathutils.Quaternion((0, -1, 0, 0))  # wxyz
    quaternion_tgt = mathutils.Quaternion((0.5, 0.5, -0.5, -0.5))

    # Compute the relative rotation matrix
    relative_rotation_matrix = compute_relative_rotation(quaternion_src, quaternion_tgt)

    # Define the center and extents of the bounding box in the camera frame
    center_cam_frame_homo = np.array([25, 0, 0, 1])
    center_world_frame_homo = camera_pose_world_frame @ center_cam_frame_homo
    center_world_frame = center_world_frame_homo[:3]

    extents = np.array([50, 50, 50])
    rotation_matrix = camera_pose_world_frame[:3, :3]

    # Create the OrientedBoundingBox
    obb = create_oriented_bounding_box(center_world_frame, rotation_matrix, extents)

    # Crop the mesh using the oriented bounding box
    cropped_mesh = crop_mesh(mesh, obb)

    # Compute the inverse transformation matrix
    world_pose_ego_frame = compute_inverse_transformation(camera_pose_world_frame)

    # Load the camera extrinsics (camera_ego_frame) from a file
    cam_ext_path = f'{processed_waymo_dataset_path}/{seq:03d}/extrinsics/0.txt'
    print(f'Loading camera extrinsics from {cam_ext_path}')
    camera_ego_frame = load_camera_pose(cam_ext_path)

    # Compute the inverse of the camera_ego_frame transformation matrix
    camera_ego_frame_inv = np.linalg.inv(camera_ego_frame)

    # Convert the world_pose_ego_frame to the camera frame
    world_pose_camera_frame = camera_ego_frame_inv @ world_pose_ego_frame

    # Convert the cropped mesh vertices to the camera frame and apply the relative rotation
    cropped_vertices = np.asarray(cropped_mesh.vertices)
    transformed_vertices = transform_vertices(cropped_vertices, world_pose_camera_frame, relative_rotation_matrix)

    # Create a new mesh with the transformed vertices
    transformed_mesh = o3d.geometry.TriangleMesh()
    transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    transformed_mesh.triangles = cropped_mesh.triangles
    transformed_mesh.vertex_colors = cropped_mesh.vertex_colors

    # Save the transformed mesh to a file
    transformed_file_name = f'{output_dir}/transformed_mesh_center_{center_cam_frame_homo[0]}_{center_cam_frame_homo[1]}_{center_cam_frame_homo[2]}_extents_{extents[0]}_{extents[1]}_{extents[2]}.obj'
    save_mesh(transformed_file_name, transformed_mesh)
    print(f'Transformed mesh saved to {transformed_file_name}')

if __name__ == '__main__':
    main()