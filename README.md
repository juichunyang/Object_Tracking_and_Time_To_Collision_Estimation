# Object_Tracking_and_Time_To_Collision_Estimation
<img src="https://github.com/CuteJui/Object_Tracking_and_Time_To_Collision_Estimation/blob/main/readme_resource/demo.png"/>
This is the second project of Sensor Fusion Nanodegree of Udacity. I fused camera and LiDAR measurements from KITTI dataset to detect, track objects in 3D space, and estimate time-to-collision. First of all, I processed image by YOLOv3 to detect and classify objects. The following figure shows the result. Based on the bounding box found by YOLOv3, I developed a way to track 3D objects over time by keypoint correspondences. Next, I used two different methods to calculate time-to-collision(TTC), LiDAR based and camera based TTC. The structure of the environment is built by the main instructor, Andreas Haja.
<img src="https://github.com/CuteJui/Object_Tracking_and_Time_To_Collision_Estimation/blob/main/readme_resource/object classification.png"/>

## LiDAR Based TTC
I projected the 3D LiDAR points of preceding car into 2D image plane by using homogeneous coordinates. The projection is shown in the following figure Next, I distributed the 3D LiDAR points to the corresponding bounding box. Finally, I computed TTC based on the closest 3D LiDAR points in the correspondence bounding box of different frames.
<img src="https://github.com/CuteJui/Object_Tracking_and_Time_To_Collision_Estimation/blob/main/readme_resource/Lidar_to_camera.png"/>

## Camera Based TTC
I used various combinations of detector/descriptor to find keypoints in each image and match these keypoints between different frames. The following figures show the keypoints detection by FAST detector and ORB descriptor as well as keypoints matching between two images. Next, I distributed these keypoints to the corresponding bounding box found by YOLOv3. Finally, I computed the TTC based on the distance of each keypoint in each frame.
<img src="https://github.com/CuteJui/Object_Tracking_and_Time_To_Collision_Estimation/blob/main/readme_resource/FAST_detector.png"/>
<img src="https://github.com/CuteJui/Object_Tracking_and_Time_To_Collision_Estimation/blob/main/readme_resource/match.png"/>

## Usage
Clone the package.
```
git clone https://github.com/CuteJui/Object_Tracking_and_Time_To_Collision_Estimation.git
```
Go to the Object_Tracking_and_Time_To_Collision_Estimation directory
```
cd /home/user/Object_Tracking_and_Time_To_Collision_Estimation
```
Create a new directory
```
mkdir build
```
Go into the build directory
```
cd build
```
Run cmake pointing to the CMakeList.txt in the root
```
cmake ..
```
Run make
```
make
```
Run the executable
```
./3D_object_tracking
```

## Detector and Descriptor 

- `detectorType` : `SHITOMASI`, `HARRIS`, `FAST`, `BRISK`, `AKAZE`, `SIFT`
	- The type of detector. 
- `descriptorType` : `BRISK`, `BRIEF`, `ORB`, `FREAK`, `AKAZE`, `SIFT`
	- The type of descriptor. 
- `matcherType` : `MAT_BF`, `MAT_FLANN`.
	- The algorithm used for matching between keypoints.
- `selectorType` : `SEL_NN`, `SEL_KNN`
	- The keypoint selection method
- `bVis` : `False` (disable), `True` (enable)
	- Enable or disable the visualization.
- `bLimitKpts` : `False` (disable), `True` (enable)
	- Limiting keypoint display results. Not recommend but useful for debug.


