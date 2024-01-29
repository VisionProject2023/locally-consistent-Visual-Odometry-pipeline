# BestVision
In this repository, a locally consistent VO pipeline is built, implemented as a Markovian, asynchronous architecture.
This project was completed as part of the Vision Algorithms for Mobile Robotics Course from Prof. Davide Scaramuzza at ETHZ and UZH (https://rpg.ifi.uzh.ch/teaching.html). 
The team consisted of (alphabetically): Zeno Hamers, Riccardo Maggioni, Augusto Mondelli and Nicola Taddei. Each team member's contributions can be seen on the GitHub page.
The full report can be viewed here: [Project Report](https://github.com/rickymaggio02/BestVision/blob/main/mini_project_VO_pipeline_Hamers_Maggioni_Mondelli_Taddei.pdf)

## Results

### KITTI dataset

![](https://github.com/rickymaggio02/BestVision/blob/main/video-results/kitti_gif.gif) 

VO Pipeline Path Reconstruction          |  Ground Truth
:-------------------------:|:-------------------------:
![](https://github.com/rickymaggio02/BestVision/blob/main/result-trajectory-plots/kitti_trajectory__sift-sift_0-2759_frames.png)  |  ![](https://github.com/rickymaggio02/BestVision/blob/main/ground-truth-trajectories/ground_truth_trajectory_kitti.png)



### Malaga dataset
![](https://github.com/rickymaggio02/BestVision/blob/main/video-results/malaga_gif.gif)

VO Pipeline Path Reconstruction          |  Ground Truth
:-------------------------:|:-------------------------:
![](https://github.com/rickymaggio02/BestVision/blob/main/result-trajectory-plots/malaga_trajectory__sift-sift_0-2119_frames.png) | ![](https://github.com/rickymaggio02/BestVision/blob/main/ground-truth-trajectories/ground_truth_trajectory_malaga.png)


### Parking dataset
![](https://github.com/rickymaggio02/BestVision/blob/main/video-results/parking_gif.gif)

VO Pipeline Path Reconstruction          |  Ground Truth
:-------------------------:|:-------------------------:
![](https://github.com/rickymaggio02/BestVision/blob/main/result-trajectory-plots/parking_trajectory__sift-sift_0-597_frames.png) | ![](https://github.com/rickymaggio02/BestVision/blob/main/ground-truth-trajectories/ground_truth_trajectory_parking.png)







## Run instructions
Before the code can be executed, the KITTI, MALAGA and parking datasets should be placed in this same repository under the name 'dataset-kitti', 'malaga-urban-dataset-extract-07' and 'parking' respectively. The datasets can be downloaded on the website of the Robotics and Perception Group (https://rpg.ifi.uzh.ch/teaching.html)

To run the VO pipeline, run python "src/main.py". Relevant parameters can be adjusted in the config.yaml file. 
The vo_pipeline.py file contains the modular subcomponents of which the Markovian VO pipeline is constructed. The visual.py file produces a 


## VO Pipeline Architecture (Markovian)
The VO pipeline is designed as a Markovian, asynchronous process. With every new frame, the state is updated. The state consists of:
1) detected keypoints 'P' 
2) the corresponding triangulated 3D landmarks 'X'

To make sure that the triangulated 3D landmarks that we add to the state are accurate enough, the baseline between the corresponding frames should be high enough. To track this, the following extended state is updated after each frame:
1) candidate keypoints 'C' (will be triangulated once this keypoint is successfully tracked for long enough) 
2) first observations of the tracked keypoint 'F'
3) camera pose at the first observation of the keypoint 'T'


### Initialization

    class VOInitializer():

    1 - Establish keypoint correspondences between 2 manually selected bootstrap frames (using feature descriptor matching, SIFT detector and descriptor preferred)

    2 - Estimate the camera pose of the second bootstrap frame (using the 5-point algorithm, 2D-2D, calibrated cameras)

    3 - Triangulate the feature correspondences as 3D landmarks and add them to the state vector

    Input: 2 frames to initialize
    Output: [S: state, Twc: transformation world-camera of the second bootstrap frame]


### Continuous operation

    class KeypointsToLandmarksAssociation():

    1 - Track the detected keypoints (and corresponding 3D landmarks) over multiple frames with the KLT  tracking algorithm (dense method)

    class PoseEstimator():

    2 - Estimate the position of the camera from the keypoint to landmark associations with the PnP RANSAC algorithm

    class LandmarkTriangulator():
    
    3 - Triangulate new 3D landmarks when the baseline threshold is met for the candidate keypoints 'C' of the extended state

Detailed documentation about the different classes (and methods contained in them) is available as comments in the vo_pipeline.py source code. The [project report](https://github.com/rickymaggio02/BestVision/blob/main/mini_project_VO_pipeline_Hamers_Maggioni_Mondelli_Taddei.pdf) can be referenced for additional clarifications.

## System specification
This code was developed on a Windows PC with the following system specifications:
 16GB of RAM, 2.4Ghz intel i7 processor (13th Gen), 64-bit

