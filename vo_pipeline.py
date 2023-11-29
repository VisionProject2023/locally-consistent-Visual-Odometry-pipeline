
import numpy as np

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

        self.K = K
        self.previous_image = np.ndarray
        self.state = {'P' : np.ndarray, 'X' : np.ndarray}
        self.candidate_keypoints = {'P' : np.ndarray, 'X' : np.ndarray, 'C' : np.ndarray,'F' : np.ndarray,'T' : np.ndarray}

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
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K
        pass

    def initialize(self, frame1: np.ndarray, frame2: np.ndarray) -> dict:
        '''
        Compute the pose of frame2 with respect to frame1 and outputs 3D landmarks

        Inputs: 
            frame1: HxW np.ndarray
            frame2: HxW np.ndarray

        Outputs:
            T: 4x4 np.ndarray representing pose
            S: dictionary with keys 'P' and 'X' representing 3D landmarks
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