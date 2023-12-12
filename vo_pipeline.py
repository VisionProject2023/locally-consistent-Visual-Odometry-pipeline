import types
import numpy as np
#import matplotlib.pyplot as plt
from typing import Dict
import cv2


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


    def get_keypoint_matches(self, frame1: np.ndarray, frame2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Match keypoints between two frames using ORB feature detector and (sparse) KLT tracking'''
        

        # Idea: the VOInitialization should be as accurate as possible, so we can use a more robust descriptor (e.g. SIFT or LIFT)
        # Idea: use ORB to find candidate keypoints, then use Tomasi corner detector

        # Initialize feauture detector and descriptor
        # Shi-Tomashi corner detector
        kps_f1 = cv2.goodFeaturesToTrack(frame1, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=None, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)
        kps_f2 = cv2.goodFeaturesToTrack(frame2, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=None, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)
        
        # SIFT corner descriptor
        sift = cv2.SIFT_create()
        des1 = sift.compute(frame1, kps_f1)
        des2 = sift.compute(frame2, kps_f2)
        
        # Feature matching (compare the descriptors, brute force)
        bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=False) # crossCheck=True is an alternative to the ratiotest (proposed by D. Lowe in SIFT paper)	
        kp_matches = bf.knnMatch(des1, des2, k=2) # k=2: return the two best matches for each descriptor

        # Apply ratio test (preventing false matches)
        good_kp_matches = []
        for m,n in kp_matches:
            if m.distance < 0.8*n.distance: # "distance" = distance function = how similar are the descriptors
                good_kp_matches.append([m])
        
        # Optional features:
        # - (computational efficiency): implement FLANN (Fast Library for Approximate Nearest Neighbors) matcher
        # - implement/use ORB detector (which is fast and rotation invariant, but not very robust to noise)
        # orb = cv2.ORB_create()
    
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

        # return the good tracking points of the old and the new frame
        
    
        return good_kp_matches

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
        
        # use CV2, triangulatePoints() to triangulate the 2D-2D correspondences
        
        
        
    def VO_initializer(self,):
       
        # first pose starts at zero 
        #pose1 = 
        
        
        # get keypoint matches
        
        # get the keypoints to 3D landmarks
        Landmarks3D =cv2.triangulatePoints(pose1, pose2, pts1, pt2)



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

        # DIRE A NICOLA DI PASSARMI cur_pose !!!
        # ATTENZIONE: costruire candidate_keypoints['T'] come un vettore di tre dimensioni (K,4,4) !!!


        # new_candidates sono i nuovi keypoints individuati da ricky nella cur_frame e NON presenti nelle frame precedenti
        # => SPIEGARE A RICKY COME FARSELI PASSARE: direi che può semplicemente passare una lista di questi new_candidates
        # che trova prendendo i keypoints di cur_frame che non hanno corrispondenza nella prec_frame 
        # ATTENZIONE: ogni volta vengono inseriti tutti in new_candidates tutti i keyframe che non sono ancora stati
        # validati, poi IO procedo a valutare se sono effettivamente nuovi o se erano già stati individuati ma NON ANCORA 
        # validati 
        
        # procedo a valutare quali di questi di new_candidates erano già stati precedentemente tracciati e quali invece sono
        # nuovi. Procedo inoltre ad eliminare i candidate_points che non sono stati re-individuati.


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

        