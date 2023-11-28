# BestVision
Best project for vision course

A Markovian, asynchronous architecture is implemented


The pipeline consists of the following segments:
a - Initialization
    1 - Establish keypoint correspondences between 2 manually selected frames (using patch matching or KLT)
    2 - Triangulate a point cloud of 3D landmarks (implement RANSAC)

    Classes:
    - class Initialization
    Input: 2 frames to initialize
    Output: [S: state, Twc: transformation world-camera]


b - Continuous operation
    1 - Estimate the position of the camera by associating keypoints in the current frame to previously triangulated 3D landmarks (P3P RANSAC)

    2 - Update 3D landmarks for keyframes (when the baseline treshold is met) 

    Classes:
    Master Class processFrame()

        - class KeypointsToLandmarksAssociation
        (Associats keypoints to existing landmarks)
        Input: previous frame I_i-1, previous state S_i-1, current frame I_i
        Output: dictionary with keypoints in frame i, associated to already identified 3D points 

        - class NextPoseEstimator
        (estimating the next pose, camera localization: P3P RANSAC)
        Input: dictionary with keypoints and associated 3D points 
        Output: pose Twc of frame i

        ----------- Determine if the baseline is higher than the threshold ---------

        - class NewLandmarksTriangulation
        (triangulates new landmarks, asynchronously)
        Input:
        Output: list of new 3D points to add
        -> put into buffer


Buffers: 
1 - the state S_i-1 from the previous frame, (to do pose estimation, maintaining Markovian principle)
2 - complete history of detected 3D landmarks (useful for mapping and loop closure)