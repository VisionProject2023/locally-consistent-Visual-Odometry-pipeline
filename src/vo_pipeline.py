import types
import numpy as np
#import matplotlib.pyplot as plt
from typing import Dict
import cv2
import yaml

# type hinting guideline: images in openCV are numpy arrays

# load yaml configurations
config_path = 'config.yaml'
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load the configuration
config = load_config('config.yaml')


class BestVision():
    '''
    This class contains the entire pipeline. We can consider adding more attributes in order to store a 
    complete map of the track as well as the complete trajectory. We can also add stuff for the 0.5 feature
    '''

    def __init__(self, K: np.ndarray):
        '''
        This method builds the object and creates the attributes we are going to use. In particular we store the last image received and a state dictionary which contains
        information about the 3D landmarks that we identified in the previous step (the state refers only to the previous step since the 
        pipeline has to be markovian). We also store the candidate_keypoints that will be used in order to generate new 3D landmarks whenever possible.
        
        Inputs:
            K: 3x3 np.ndarray of intrinsic parameters

        '''

        self.K = K # Intrinsic parameters, the camera is assumed to be calibrated
        self.previous_image = np.ndarray
        self.state = {'P' : np.ndarray, 'X' : np.ndarray}
        self.state_with_candidate_keypoints = {'P' : np.ndarray, 'C' : np.ndarray,'F' : np.ndarray,'T' : np.ndarray}

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
    
    The aim of the VOInitialization should be to estimate the pose of frame2 and 3D landmarks as accurate as possible

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

    def get_keypoint_matches(self, frame1: np.ndarray, frame2: np.ndarray) -> (np.array, np.array):
        ''' 
        Match keypoints between two frames using different detectors (default = SIFT) and the SIFT descriptor
        
        Args:
            frame1: HxW np.ndarray
            frame2: HxW np.ndarray
        
        Returns:
            good_kps_f1: list of keypoints in frame1
            good_kps_f2: list of keypoints in frame2
        '''

        if config['init_detector_descriptor'] =='shi-tomasi-sift':
            
            # Initialize feauture detector and descriptor
            # Shi-Tomashi corner detector, to test and figure out (will probably have to change the type of kps_f1 to cv.KeyPoint)
            kps_f1 = cv2.goodFeaturesToTrack(frame1, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=None, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)
            kps_f2 = cv2.goodFeaturesToTrack(frame2, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=None, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)
            
            # Convert keypoints to cv2.KeyPoint format
            kps_f1 = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in kps_f1]
            kps_f2 = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in kps_f2]
            
            # SIFT corner descriptor
            sift = cv2.SIFT.create()
            des1 = sift.compute(frame1, kps_f1) # describes the features in frame 1
            des2 = sift.compute(frame2, kps_f2) # describes the features in frame 2
            
            # des1 and des2 don't have the correct format here yet to be processed by bf.knnMatch
            # this has to be figured out!
            
        if config['init_detector_descriptor'] == 'sift':
            # All SIFT implementat  ion:
            sift = cv2.SIFT.create()
            kps_f1, des1 = sift.detectAndCompute(frame1, None)
            kps_f2, des2 = sift.detectAndCompute(frame2, None)
            
        if config['init_detector_descriptor'] == 'orb':
            # All ORB implementation:
            orb = cv2.ORB.create()
            kps_f1, des1 = orb.detectAndCompute(frame1, None)
            kps_f2, des2 = orb.detectAndCompute(frame2, None)
        
        # Feature matching (compare the descriptors, brute force)
        bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=False) # crossCheck=True is an alternative to the ratiotest (proposed by D. Lowe in SIFT paper)	
        kp_matches = bf.knnMatch(des1, des2, k=2) # k=2: return the two best matches for each descriptor

        # Apply ratio test (preventing false matches)
        good_kp_matches = []
        for m,n in kp_matches:
            if m.distance < 0.8*n.distance: # "distance" = distance function = how similar are the descriptors
                good_kp_matches.append(m)
                       
        # convert kp matches to lists
        good_kps_f1 = np.array([kps_f1[match.queryIdx].pt for match in good_kp_matches])
        good_kps_f2 = np.array([kps_f2[match.trainIdx].pt for match in good_kp_matches])
  
        # return the good tracking points of the old and the new frame
        return good_kps_f1, good_kps_f2
        
        
        # # Optional KLT implementation
        # kps_f1_KLT = cv2.KeyPoint_convert(kps_f1) # Convert to Point2f format for KLT tracker
        
        # # The KLT tracker will match the keypoints of the first frame with corresponding keypoints of the second frame
        # # Parameters for KLT tracker
        # lk_params = dict(winSize=(15, 15),
        #                  maxLevel=2,
        #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # # Applying KLT tracker
        # kps_f2_KLT, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, kps_f1_KLT, None, **lk_params)

        # # Filter out good points
        # good_keypoints_new_frame = kps_f1_KLT[st]
        # good_keypoints_old_frame = kps_f2_KLT[st]
        

    def get_pose_estimate(self, kps_f1: list, kps_f2: list) -> np.ndarray:
        '''
        Estimatea the pose of the second frame based on the 2D-2D correspondences between the two frames (5-point RANSAC algorithm)
        The pose is always relative to the very first frame (which is the world frame)
        
        Args: 
            kps_f1: list of keypoints in frame1
            kps_f2: list of keypoints in frame2
            
        Returns: 
            T_hom: 4x4 np.ndarray (homogeneous transformation matrix) representing the pose of frame2 with respect to frame1
        '''
        
        # Use the 5-point RANSAC to compute the essential matrix, taking outliers into account (5-point algorithm uses the epipolar geometry to compute the essential matrix)
        # But, the 5-point algorithms can return up to 10 solution of the essential matrix

        # Compute the essential matrix using the RANSAC 5-point algorithms
        E, mask = cv2.findEssentialMat(kps_f1, kps_f2, self.K, cv2.RANSAC, 0.999, 1.0, None)

        # Given the essential matrix, 4 possible combinations of R and T are possile, but only one is in front of the camera (cheirality constraint)
        _, R, t, _ = cv2.recoverPose(E, kps_f1, kps_f2, self.K, mask=mask) #recoverPose enforces the 

        # Output transformation matrix (not homogeneous!)
        return np.hstack((R, t))


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
        #function to use cv2.calcOpticalFlowPyrLK()
        #input: 
        #       prevImage
        #       nextImage
        #       prevPts -> vector of 2d points to track
        #Output:
        #       nextPts -> vector of 2D points containing the calculated new 
        #                  positions of input features in the second image
        #       status -> vector with 1 if corresponding feature has been found, 0 if not
        #       error -> output vector of errors

        next_points, status, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, state['P'])

        #remove outliers
        #we are seeing a car like vehichle, so we can exploit the 1 point ransac:
        # I imagine a 2 x N array
        #thetas should be a 1 x N array
        #paper scaramuzza: https://rpg.ifi.uzh.ch/docs/IJCV11_scaramuzza.pdf
        thetas = -2 * np.arctan((next_points[0,:]-state['P'][0,:])/(next_points[1,:]-state['P'][1,:]))
        #we generate all the possible thetas, and then generate an histogram
        hist = np.histogram(thetas)
        theta_max = np.median(hist)
        R = np.array([np.cos(theta_max), - np.sin(theta_max), 0],
                     [np.sin(theta_max),   np.cos(theta_max), 0],
                     [0 ,                0,                   1])
        #the paper (Scaramuzza) says that I can set rho to  1, see if it make sense with the reprojected points
        T =np.array([np.cos(theta_max/2), np.sin(theta_max/2), 0]).T
        #reprojection error:
        projected_points = (np.vstack[(R,T)] @ np.vstack((state['P'], np.ones_like(state['P'].shape[0]))))[:,0:2]
        error_threshold = 1 #error threshold of one pixel
        filter = next_points[np.linalg.norm(next_points - projected_points )< error_threshold]

        #return new status and connection
        new_P = state['X'][status]
        new_P_error_free = new_P[filter]
        new_state = {'P': new_P_error_free, 'X':next_points[filter]}

        return new_state


class PoseEstimator():
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K

        """ Constants """
        self.REPOJ_THRESH = 3
    
    def estimatePose(self, associations: Dict[np.ndarray,np.ndarray]) -> np.ndarray:
        '''
        Computes the current pose from the associations found in previous steps

        Inputs:
            associations: dictionary with keys 'P' and 'X_old' that contain 2D points from the new frame and the corresponding matching in the state vector

        Outputs:
            T: 4x4 np.ndarray representing the pose of the new frame with respect to the world frame
        '''
        success, R, t, inliers = cv2.solvePnpRansac(objectPoints = associations['X_old'], 
                                  imagePoints = associations['P'],
                                  cameraMatrix = self.K,
                                  distCoeffs = None,
                                  flags=cv2.SOLVEPNP_P3P,
                                  confidence=0.9999 ,
                                  reprojectionError=self.REPOJ_THRESH)
        
        T = np.concatenate([np.concatenate([R,t], axis=-1),np.array([0,0,0,1])], axis=0)
        return T


        

class LandmarkTriangulator():
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K
        pass

    def triangulateLandmark(self, candidate_keypoints: dict, old_frame: np.ndarray, cur_frame: np.ndarray, features: np.ndarray, state, new_candidates_list, cur_pose) -> dict:
        '''
        Inputs:
            candidate_keypoints: dict as defined in the main class
            frame: HxW np.ndarray
            features: 2xN np.ndarray containing 2D points detected in the last frame

        Output:
            new_landmarks: dictionary with keys 'P' and 'X'  for the new identified landmarks
        '''

        # TELL NICOLA TO PASS ME cur_pose !!!
        # ATTENTION: construct candidate_keypoints['T'] as a three-dimensional vector (K,4,4) !!!

        # new_candidates are the new keypoints identified by Ricky in the cur_frame and NOT present in the previous frames
        # => EXPLAIN TO RICKY HOW TO PASS THEM: I would say he can simply pass a list of these new_candidates
        # that he finds by taking the keypoints of cur_frame that have no correspondence in the prec_frame
        # ATTENTION: every time all the keyframes that have not yet been
        # validated are inserted in new_candidates, then I proceed to evaluate if they are actually new or if they had already been identified but NOT YET
        # validated

        # I proceed to evaluate which of these new_candidates had already been previously tracked and which are
        # new. I also proceed to remove the candidate_points that have not been re-identified.ovi. Procedo inoltre ad eliminare i candidate_points che non sono stati re-individuati.


        # TRACK CANDIDATE KEYPOINTS
        p0 = candidate_keypoints['C']
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, cur_frame, p0, None)
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        

        # REMOVE CANDIDATE KEYPOINTS THAT HAVE NOT BEEN RETRACKED
        bad_old = p0[st==0]
        # Create a boolean mask to identify the positions of elements to be removed
        mask = np.isin(candidate_keypoints['C'], bad_old)
        # Get the indices of elements to be removed
        indices_to_remove = np.where(mask)[0]
        # Check if any elements are found before attempting removal
        if indices_to_remove.size > 0:
            # Remove the elements using boolean indexing
            candidate_keypoints['C'] = np.delete(candidate_keypoints['C'], indices_to_remove)
            candidate_keypoints['F'] = np.delete(candidate_keypoints['F'], indices_to_remove)
            candidate_keypoints['T'] = np.delete(candidate_keypoints['T'], indices_to_remove)


        # ADD NEW CANDIDATE KEYPOINTS THAT HAVE NEWLY BEEN TRACKED
        new_candidates = {}
        new_candidates['C'] = new_candidates_list
        new_candidates['F'] = new_candidates_list
        new_candidates['T'] = np.stack([(np.eye(4) @ cur_pose) for _ in range(len(new_candidates_list))])

        candidate_keypoints['C'] = np.concatenate((candidate_keypoints['C'], new_candidates['C']))
        candidate_keypoints['F'] = np.concatenate((candidate_keypoints['F'], new_candidates['F']))
        candidate_keypoints['T'] = np.concatenate((candidate_keypoints['T'], new_candidates['T'])) 
        

        # UPDATE C OF CANDIDATE POINTS THAT HAVE BEEN RETRACKED
        # Create a boolean mask to identify the positions of elements to be removed
        mask = np.isin(candidate_keypoints['C'], good_old)
        # Get the indices of elements to be removed
        candidate_keypoints['C'][mask] = good_new
        

        # VALIDATE NEW POINTS
  
        pass

        