import numpy as np
import json
from scipy.spatial.transform import Rotation  

# Path where matrix should be stored
save_path = "data/markers"
tag_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def get_aruco_data(json_file, tag_id):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    translations = data.get("AruCo-Tag-Positionen (Tag-Mitte)", {})
    rotations = data.get("AruCo-Tag-Rotationen um x, y, z", {})
    
    trans_vector = np.array(translations.get(str(tag_id), []))
    rot_vector = np.array(rotations.get(str(tag_id), []))
    
    return trans_vector, rot_vector

# We need transformation matrix to transform the coordinates of one object to the coordinates of another object
# The function combines the rotation and the transformation matrix
# Generate transforamtion matrix that transforms a vector which is in Demonstrator KoSy to ArUco KoSy
def build_transformation_matrix(json_path, print=False, save=True):
    for tag_id in tag_ids:
        # First, build an identity matrix 4x4
        transformation_matrix = np.eye(4)
        #                    The row marked with * isn't used
        #     //////|        The rotation matrix is placed on the rows and columns marked with //  
        # // [1 0 0 0] -     The translation vector is placed on the rows and column marked with - or |        
        # // [0 1 0 0] -     Combining them we receive a transformation matrix
        # // [0 0 1 0] -     The rotation matrix looks like
        # *  [0 0 0 1]*       [x y z] 
        #                     [i j k]
        #                     [m n l]     
        #     
        #                    The translation vector looks like
        #                     [x]
        #                     [y]
        #                     [z]
        #    The last row of the matrix [0 0 0 1] can remain like this. We wont use it (we dont perform rotatation or translation in 4 dimension :()                  
        
        # Transform euler angles to rotation matrix
        trans_vector, rot_vector = get_aruco_data(json_path, tag_id)
        r = Rotation.from_euler("xyz", rot_vector,degrees=True)
        rotation_matrix = r.as_matrix()

        # Add components to matrix
        transformation_matrix[:3, :3] = np.array(rotation_matrix)
        transformation_matrix[:3, 3] = trans_vector

        # Save
        if save:
            filename = "/ID" + str(tag_id) + ".npy"
            np.save(save_path + filename, transformation_matrix, allow_pickle=False)

        if print:
            print(transformation_matrix)

if __name__ == "__main__":
   build_transformation_matrix('tag_info.json')
