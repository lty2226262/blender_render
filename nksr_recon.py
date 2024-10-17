import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud
import torch
from pycg import vis
import nksr

def load_waymo_example(input_file):
    """
    Load the Waymo example point cloud data.

    Returns:
        xyz (np.ndarray): Array of point coordinates.
        sensor (np.ndarray): Array of sensor positions.
    """
    waymo_path = Path(input_file)

    # Load point cloud data using PyntCloud
    pcloud = PyntCloud.from_file(str(waymo_path))
    pdata = pcloud.points

    # Extract point coordinates and sensor positions
    xyz = np.stack([pdata['x'], pdata['y'], pdata['z']], axis=1)
    sensor = np.stack([pdata['sensor_x'], pdata['sensor_y'], pdata['sensor_z']], axis=1)
    return xyz, sensor

def main():
    """
    Main function to parse arguments and process the static map.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input point cloud file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output mesh file')
    args = parser.parse_args()

    # Load point cloud data
    xyz_np, sensor_np = load_waymo_example(args.input_file)

    # Set up the device for computation
    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")

    # Convert numpy arrays to PyTorch tensors and move to the device
    input_xyz = torch.from_numpy(xyz_np).float().to(device)
    input_sensor = torch.from_numpy(sensor_np).float().to(device)

    # Perform reconstruction
    field = reconstructor.reconstruct(
        input_xyz, sensor=input_sensor, detail_level=None,
        # Minor configs for better efficiency (not necessary)
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
        # Chunked reconstruction (if OOM)
        chunk_size=25.6,
        preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0)
    )
    
    mesh = field.extract_dual_mesh(mise_iter=1)
    mesh = vis.mesh(mesh.v, mesh.f)

    # Save the mesh to the specified output file
    o3d.io.write_triangle_mesh(args.output_file, mesh)
    vis.show_3d([mesh], [vis.pointcloud(xyz_np)])

if __name__ == '__main__':
    main()