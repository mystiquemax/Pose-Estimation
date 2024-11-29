# Pose Estimation project

The entire project focuses on detecting ArUco markers placed on an object in a lab at my university. Images and scans of the object are captured, and the ArUco tags are detected. Based on the detected ArUco tags, the object is then transformed into the scan coordinate system. Finally, shifting is applied to compensate for the depth estimation issues of the scanner.

# Files in the repository

* **detect.py**: The functions in this file detect the ArUco tags on the images
* **generate_markers.py**: The function here generates the ArUco tags
* **tag_info.json**: In this file, the size of the object is defined, on which ArUco tags are placed and also the positions of the ArUco tags based on the object's coordinate system.  
* **matrix_to_numpy.py**: The functions here create transformation matrix from the coordinate system of the demonstrator to the coordinate system of  a given ArUco tag and save it.
* **transformation.py**: The functions in this file make the transformation from coordinate system of the demonstrator to the coordinate system of the point cloud. I added a *.ipynb* file to show the results. Detailed description is also added.
* **pcd_shift.py**: Because of the depth estimation problem, addition shifting of the point cloud was necessary, so it's aligned with the CAD-model. I added a *.ipynb* file to show the results. Detailed description is also added.
* **utils.py**: The functions here provided the ArUco type dictionary and safe additional data.

**Note**: For privacy reasons, the folder "data" is not being published.
