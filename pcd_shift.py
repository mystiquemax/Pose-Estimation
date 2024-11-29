import open3d as o3d
import detect
import numpy as np
import glob
import cv2
import json

# Marker size in meter:  0.105
# 0,0,0 means the back of the arUco points to the floor and the Kosy of the Demostrator 

depth_directions_of_arucos = {"x": [5,8],
                              "y": [1,2,3,4,6,7,10], 
                              "z": [9]}

# Demostrator
def get_mesh(path): 
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    return mesh

# Pointcloud
def get_path_to_pcd(path_to_scan):
    return glob.glob(path_to_scan + '*.ply')[0]

def get_pointcloud(path):
    point_cloud = o3d.io.read_point_cloud(path)
    return point_cloud

# Image
def get_path_to_image(path_to_scan):
    return glob.glob(path_to_scan + '*.png')[0]

def find_nearest_point_in_direction(pcd, reference_point, direction_vector):
    """
    Get the point that is nearest to the 'reference_point' in the pointcloud in the given direction

    Args: pcd = pointcloud from which we take the points
          reference point = the starting point from which we measure the direction
          direction_vector = the direction in which we will search the point
    Return: nearest point to the reference point   
    """

    # Normalize the direction vector (# in our case the vector is already normalized. this is for any case if we use some other direction thats not on the axis (e.g if the demostrator was hexagonal prism)
    direction_unit = direction_vector / np.linalg.norm(direction_vector)
    
    # Calculate the vector from the center of the Aruco to each point in the point cloud
    vectors_to_points = np.asarray(pcd.points) - reference_point
    
    # Project these vectors onto the direction unit vector
    projections = np.dot(vectors_to_points, direction_unit)
    
    # Calculate the distances from the projected points to the original points and find the eucliand distance to the point
    distances = np.linalg.norm(vectors_to_points - np.outer(projections, direction_unit), axis=1)
    
    # Find the index of the nearest point based on the minimum distance
    nearest_index = np.argmin(distances)
    
    # Get the nearest point
    nearest_point = pcd.points[nearest_index]
    
    return nearest_point


def get_mean_translation_vector_y(pcd, ids):
    divide = 0

    mean_translation_vector_y = np.array([.0, .0, .0])
   
    for i in ids.flatten():
      if i in depth_directions_of_arucos["y"]:
         with open('tag_info.json', 'r') as f:
            data = json.load(f)
            translation = data["AruCo-Tag-Positionen (Tag-Mitte)"][str(i)]
            translation = np.array([translation])
            aruco_center = o3d.geometry.PointCloud()
            aruco_center.points = o3d.utility.Vector3dVector(translation)
            direction_vector = [.0, 1.0, .0]
            nearest_point = find_nearest_point_in_direction(pcd, translation, direction_vector)
            ####################
            # All arUco tags with depth direction y-axis are only on the left side of the demonstrator.
            # That's why this is enough.
            shift_distance = np.abs(translation[0][1] - nearest_point[1])
            ####################
            translation_vector = np.array([.0, shift_distance, .0])
            print("ArUco Tag number: " + str(i) + " with depth direction in y axis")
            print("Translation vector of the give ArUco Tag",translation_vector)
            mean_translation_vector_y += translation_vector
            divide+=1

    if divide != 0:
       mean_translation_vector_y /= divide

    return mean_translation_vector_y   

def get_mean_translation_vector_x(pcd,ids):  
    divide = 0  
    
    mean_translation_vector_x = np.array([.0, .0, .0])

    for i in ids.flatten():
      if i in depth_directions_of_arucos["x"]:
         with open('tag_info.json', 'r') as f:
            data = json.load(f)
            translation = data["AruCo-Tag-Positionen (Tag-Mitte)"][str(i)]
            translation = np.array([translation])
            aruco_center = o3d.geometry.PointCloud()
            aruco_center.points = o3d.utility.Vector3dVector(translation)
            direction_vector = [1.0, .0, .0]
            nearest_point = find_nearest_point_in_direction(pcd, translation, direction_vector)
            ####################
            # We have arUco tag only on the front size of the demonstrator.
            # That's why this is enought.
            shift_distance = translation[0][0] - nearest_point[0]
            ####################
            translation_vector = np.array([shift_distance, .0, .0])
            print("ArUco Tag number: " + str(i) + " with depth direction in x axis")
            print("Translation vector of the give ArUco Tag",translation_vector)
            mean_translation_vector_x += translation_vector
            divide+=1

    if divide != 0:
       mean_translation_vector_x /= divide

    return mean_translation_vector_x   

def get_mean_translation_vector_z(pcd,ids):  
    divide = 0  
    
    mean_translation_vector_z = np.array([.0, .0, .0])

    for i in ids.flatten():
      if i in depth_directions_of_arucos["z"]:
         with open('tag_info.json', 'r') as f:
            data = json.load(f)
            translation = data["AruCo-Tag-Positionen (Tag-Mitte)"][str(i)]
            translation = np.array([translation])
            aruco_center = o3d.geometry.PointCloud()
            aruco_center.points = o3d.utility.Vector3dVector(translation)
            direction_vector = [.0, .0, 1.0]
            nearest_point = find_nearest_point_in_direction(pcd, translation, direction_vector)
            ####################
            # We have only one arUco at the top of the demonstrator.
            # That's why this is enough.
            shift_distance = translation[0][2] - nearest_point[2]
            ####################
            translation_vector = np.array([.0, .0, shift_distance])
            print("ArUco Tag number: " + str(i) + " with depth direction in z axis")
            print("Translation vector of the give ArUco Tag", translation_vector)
            mean_translation_vector_z += translation_vector
            divide+=1

    if divide != 0:
       mean_translation_vector_z /= divide

    return mean_translation_vector_z   


if __name__ == "__main__": 
    path_to_mesh = "data/demonstrator_model/scaled_demonstrator.stl"
    path_to_pcd = "data/scans_roughly_aligned/10_Mean.ply"
    path_to_image = "data/Mech_Eye/2024-01-19/10/"

    # load the CAD-file
    mesh = get_mesh(path_to_mesh)
    # load the point cloud
    pcd = get_pointcloud(path_to_pcd)
  
    image = cv2.imread(get_path_to_image(path_to_image))
    corners, ids, _ = detect.detect_all_markers(image, "DICT_4X4_100")
    print("Detected ArUcos in the image:", ids.flatten())
    
    mean_translation_vector = get_mean_translation_vector_x(pcd, ids) + get_mean_translation_vector_y(pcd, ids) + get_mean_translation_vector_z(pcd,ids)
    print("Mean translation vector:", mean_translation_vector)
    pcd = pcd.translate(mean_translation_vector)

    pcd_original = get_pointcloud(path_to_pcd)   
    o3d.visualization.draw_geometries([mesh, pcd_original], window_name="Demostrator and Pointcloud Mean", width=800, height=600) 
    o3d.visualization.draw_geometries([mesh, pcd], window_name="Demostrator and Pointcloud", width=800, height=600)         

 
   
