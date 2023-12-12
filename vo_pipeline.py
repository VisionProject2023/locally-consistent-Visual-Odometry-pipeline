import types
import numpy as np
import cv2
#from matplotlib import pyplot as plt


class BestVision():
    '''
    This class contains the entire pipeline. We can consider adding more attributes in order to store a 
    complete map of the track as well as the complete trajectory. We can also add stuff for the 0.5 feature
    '''

    def __init__(self, K: np.ndarray):
        '''
        This method builds the object and creates the attributes the we are going to use. In particular we store the last image received and a state dictionary which contains
        information about the 3D landmarks that we identified in the previous step (the state refers only to the previous step since the 
        pipeline has tobe markovian). We also store the candidate_keypoints that will be used in order to generate new 3D landmarks whenever possible.
        
        Inputs:
            K: 3x3 np.ndarray of intrinsic parameters

        '''

        self.K = K # Intrinsic parameters, the camera is assumed to be calibrated
        self.previous_image = np.ndarray
        self.state = {'P' : np.ndarray, 'X' : np.ndarray}
        self.candidate_keypoints = {'P' : np.ndarray, 'C' : np.ndarray,'F' : np.ndarray,'T' : np.ndarray}

    def initialize(frame_sequence: np.ndarray) -> np.ndarray:
        '''
        Takes as input a sequence of frames, initializes the state and candidate_keypoints and returns the configuration of the second keyframe 
        with respect to the first frame which configuration is considered as the world frame. This function makes the choice of which frame to use 
        as second frame in initialization (for example the third frame in the sequence for th KITTY dataset as suggested)

        Inputs: 
            frame_sequence: List[ HxW np.ndarray ] a list of all the frames (or just the first n)

        Outputs:
            T: 4x4 np.ndarray representing the transformation between the first frame and the second keyframe (for example the third in the image flow) 
        '''

        pass

    def processFrame(new_frame: np.ndarray) -> np.ndarray:
        '''
        Takes as input a new frame and computes the pose of this frame. It also updates the stored 3D landmarks.

        Inputs:
            new_frame: HxW np.ndarray

        Outputs:
            T: 4x4 np.ndarray which encodes the new pose with respect to the world frame
        '''
        pass


class VOInitializer():
    
    '''
    Compute the pose of frame2 with respect to frame1 and outputs 3D landmarks

    Inputs: 
        frame1: HxW np.ndarray
        frame2: HxW np.ndarray

    Outputs:
        T: 4x4 np.ndarray representing pose
        S: dictionary with keypoints 'P' (keys) and 3D landmarks 'X' (values)
    '''

    
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K


    def detect_corresponding_keypoints(self, frame1: np.ndarray, frame2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ''' Match keypoints between two frames using ORB feature detector and (sparse) KLT tracking'''
        

        # Idea: the VOInitialization should be as accurate as possible, so we can use a more robust descriptor (e.g. SIFT or LIFT)
        # Idea: use ORB to find candidate keypoints, then use Tomasi corner detector

        # Initialize feauture detector and descriptor
        # Shi-Tomashi corner detector
        kps_f1 = cv2.goodFeaturesToTrack(frame1, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=None, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)
        kps_f2 = cv2.goodFeaturesToTrack(frame2, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=None, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)
        
        # SIFT corner descriptor
        sift = cv2.SIFT_create()
        sift.compute(frame1, kps_f1)
        sift.compute(frame2, kps_f2)
        
        # Feature matching (compare the descriptors)
        

        #optional: implement/use ORB detector (which is fast and rotation invariant, but not very robust to noise)
        # orb = cv2.ORB_create()
    
        # Optional KLT implementation
        kps_f1_KLT = cv2.KeyPoint_convert(kps_f1) # Convert to Point2f format for KLT tracker
        
        # The KLT tracker will match the keypoints of the first frame with corresponding keypoints of the second frame
        # Parameters for KLT tracker
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Applying KLT tracker
        kps_f2_KLT, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, kps_f1_KLT, None, **lk_params)

        # Filter out good points
        good_keypoints_new_frame = kps_f1_KLT[st]
        good_keypoints_old_frame = kps_f2_KLT[st]

        # return the good tracking points of the old and the new frame
        return good_keypoints_old_frame, good_keypoints_new_frame

    def estimate_pose(self, kps_f1, kps_f2) -> tuple[np.ndarray, np.ndarray]:
        '''

        # Now we have a set of 2D-2D correspondences between the two frames
        # -> apply the 5-point algorithm (implemented by OpenCV) to compute the essential matrix

        # The fundamental matrix (non-calibarated cameras) and essential matrix (calibrated cameras)
        # encode the relationship between two different camera views of the same scene

        # The essential matrix encodes the relative pose (rotation and translation) between the two camera views)

        # The pose is always relative to the very first frame (which is the world frame)
        '''
        
    
        # Use the 1-point RANSAC to remove the intial outliers (with a relatively big error treshold), now we have a low rate of outliers
        # Then, use the 8-point RANSAC to remove the remaining outliers (only one solution, more robust against noise, around the same computational speed as 5-point with a low amount of outliers)
        
        # Compute the essential matrix using the RANSAC 5-point algorithms (potentially 8-point to be more robust against noise)
        E, mask = cv2.findEssentialMat(kps_f1, kps_f2, self.K, cv2.RANSAC, 0.999, 1.0, None)

        # Decompose the essential Matrix in possible rotations and translations (there are 4 possible solutions!)
        _, R, t, _ = cv2.recoverPose(E, kps_1, kps_f2, self.K)

        # Form the transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T #pose (both rotation and translation)

    def get_keypoints_to_3D_landmarks(self, frame1: np.ndarray, frame2: np.ndarray, pose: np.ndarray) -> dict:
        '''
        # Now we have the pose of the second frame with respect to the first frame
        # -> triangulate the 2D-2D correspondences to obtain 3D points

        # The 3D points are in the world frame
        '''
        
        
        pass



class KeypointsToLandmarksAssociator():
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K
        pass

    def associateKeypoints(self, old_frame: np.ndarray, new_frame: np.ndarray, state: dict, features: list) -> dict:
        '''
        Associate keypoints from old image to features of the new image.

        Inputs:
            old_frame: HxW np.ndarray
            new_frame: HxW np.ndarray
            state: dict with keys 'P' and 'X'
            features: list of 2D points corresponding to features detected in the new image

        Outputs:
            associations: dictinary with keys 'P' and 'i'. associations['P'] contains 2D points from new_frame associated with previous landmarks
                          and associations['i'] contains list of indices to which the points are associated
        '''
        pass



class PoseEstimation():
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K
        pass
    
    def estimatePose(self, associations: dict) -> np.ndarray:
        '''
        Computes the current pose from the associations found in previous steps

        Inputs:
            associations: dictionary with keys 'P' and 'i' that contain 2D points from the new frame and the corresponding matching in the state vector

        Outputs:
            T: 4x4 np.ndarray representing the pose of the new frame with respect to the previous one
        '''
        pass

class LandmarkTriangulator():
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K
        pass

    def triangulateLandmark(self, candidate_keypoints: dict, frame: np.ndarray, features: np.ndarray) -> dict:
        '''
        Inputs:
            candidate_keypoints: dict as defined in the main class
            frame: HxW np.ndarray
            features: 2xN np.ndarray containing 2D points detected in the last frame

        Output:
            new_landmarks: dictionary with keys 'P' and 'X'  for the new identified landmarks
        '''
        pass