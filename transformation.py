import os, sys
sys.path.append('./')
import open3d as o3d
import numpy as np
import cv2
import detect
import glob
from scipy.spatial.transform import Rotation  

# Mesh
def get_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals() 
    mesh.scale(0.01, center=(0, 0, 0))
    return mesh

# Point cloud  
def get_pointcloud(path, downsample=False):
    pcd = o3d.io.read_point_cloud(path)

    if downsample:
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
    return pcd

# Transform point cloud based on one arUco marker
def transform_pcd(pcd, id, path_image):
    # Get transformation matrix from Aruco marker coordinate system to Demonstrator coordinate system
    path_markers = "/data/markers/ID" 
    path_matrix = path_markers + str(id) + ".npy"
    T_D_A = np.load(path_matrix) ## Matrix that transforms vector / coordinates from demonstrator co. sy. to aruco co. sy.

    # The scan (i.e. point cloud) and the camera are in the same coordinate system
    T_Cam_Scan = np.eye(4)

    # Get rotation matrix and translation vector of Aruco marker relative to the RGB camera
    arucos = detect.get_marker_poses(image_path = path_image,
                                     tag_type="DICT_4X4_100",
                                     show=False)
    for aruco in arucos:
        if aruco['id'] == id:
            rotM, jac = cv2.Rodrigues(aruco['rvec'][0][0])
            tvec = aruco['tvec'][0][0]
            any = True

    # If the arUco is on the image taken by the camera
    if any:
        # Tvec is translation of Aruco marker relative to the RGB camera, we need it the other way round 
        tvec = np.array(tvec)
        tvec = -np.transpose(rotM).dot(tvec) #  i.e cameraPosition = -np.matrix(rotM).T * np.matrix(tvec)

        # Merge transponed rotation matrix and tvec to transformation matrix 
        T_A_Cam = np.eye(4)
        T_A_Cam[:3, :3] = np.transpose(np.array(rotM))
        T_A_Cam[:3, 3] = np.array(tvec)

        # Get complete transformation Demonstrator --> ArUco marker --> RGB camera --> Scan (i.e point cloud)
        transM = T_D_A @ T_A_Cam @ T_Cam_Scan # Takes the coordinates of a vector in demostrator coordinate system and transforms them in scan (pointcloud) coordinate system

        # Apply transformation matrix
        pcd = pcd.rotate(transM[:3, :3], center=(0, 0, 0))
        pcd = pcd.translate(transM[:3, 3])

        return pcd
    
    else:
        return 'ArUco marker with ID {id} not on image'

def get_mean_transformation_matrix_of_detected_arUcos(path_image, ids):

    path_markers = "data/markers/ID"   
   
    T_Cam_Scan = np.eye(4) 

    # Get pose of all arucos on image
    rotations = []
    translations = []
    arucos = detect.get_marker_poses(image_path = path_image,
                           tag_type="DICT_4X4_100",
                           show=False)

    for aruco in arucos:
        if aruco['id'] in ids:
            id = aruco['id']
            rotM, jac = cv2.Rodrigues(aruco['rvec'].flatten())
            tvec = aruco['tvec'].flatten()

            # Get transformation matrix D (Demostrator) to given arUco-Tag (Ai)
            path_matrix = path_markers + str(id) + ".npy"
            T_D_Ai = np.load(path_matrix) 

            # Tvec translates from Aruco CoSy to Camera CoSy, we need it the other way round
            tvec = np.array(tvec)
            tvec = -np.transpose(rotM).dot(tvec)

            # Merge transponed rotation matrix and tvec to transformation matrix
            T_Ai_Cam = np.eye(4)
            T_Ai_Cam[:3, :3] = np.transpose(np.array(rotM))
            T_Ai_Cam[:3, 3] = np.array(tvec)

            # Get complete transformation Demonstrator --> ArUco marker --> RGB camera --> Scan (i.e point cloud)
            transM = T_D_Ai @ T_Ai_Cam @ T_Cam_Scan

            # Get rotation in Euler and translation for building the mean
            rotVector = Rotation.from_matrix(transM[:3,:3]) # rotation vector from rotation marix
            rotations.append(rotVector.as_euler('xyz', degrees=True))
            translations.append(np.array(transM[:3, 3]))

    # Calculate mean of all rotations and translations and transform back to matrix
    rotArray = np.array(rotations).T
    transArray = np.array(translations).T
    mean_rot = np.mean(rotArray, axis=1)
    mean_trans = np.mean(transArray, axis=1)

    r = Rotation.from_euler("xyz", mean_rot, degrees=True)
    transM[:3, :3] = r.as_matrix()
    transM[:3, 3] = mean_trans
    return transM

def get_mean_ply(pcd, path_image, ids):
    transM = get_mean_transformation_matrix_of_detected_arUcos(path_image, ids)
    pcd.transform(transM)
    
    return pcd

def visualize(path_scan, id, mode):
    # Get pointcloud and image path from scan path
    path_mesh = 'data/demonstrator_model/demonstrator_gesamt.stl'

    path_image = glob.glob(path_scan + '*.png')[0]

    path_pcd = glob.glob(path_scan + '*.ply')[0]
    
    path_save = "data/pointclouds_transformed/"
    folder_scan = os.path.basename(os.path.normpath(path_scan))

    # Import mesh and pointcloud
    mesh = get_mesh(path_mesh)
    pcd = get_pointcloud(path_pcd)
    # Visualize the original pcd and the mesh
    o3d.visualization.draw_geometries([mesh, pcd])
    
    # Calculate and add transformation on pointcloud from a single aruco
    if mode.lower() == "single":
        pcd_new = transform_pcd(pcd, id, path_image)
        if pcd_new != 'ID not on image':
            o3d.io.write_point_cloud(path_save + folder_scan + "_ID" + str(id) + ".ply", pcd_new)

    # Using the mean value of the positions of all the arUcos        
    elif mode.lower() == "mean":
        pcd_new = get_mean_ply(pcd, path_image, ids=id)
        o3d.io.write_point_cloud(path_save + folder_scan + "_Mean.ply", pcd_new)
    
    # Plot mesh, pointcloud and additional points/lines
    if pcd_new != 'ID not on image': 
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.5, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([mesh, pcd_new, mesh_frame])



if __name__ == 'main':
    path_prefix = "data/Mech_Eye/2024-01-22/"
    mode = "mean"

    # right side arUcos
    id = [5, 6, 7, 8]

    folders = os.listdir(path_prefix)

    for folder in folders:
        path_scan_folder = path_prefix + f"{folder}/"
        visualize(path_scan_folder, id, mode)
