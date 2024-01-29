import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import cv2
import yaml
import os

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
        
        Inputs:
            K: 3x3 np.ndarray of intrinsic parameters

        '''

        self.K = K # Intrinsic parameters, the camera is assumed to be calibrated
        self.previous_image = np.ndarray # last received image
        self.state = {'P': np.ndarray, 'X': np.ndarray} # 'P' (keypoints), 'X' (3D landmarks) (state refers to the previous step since the pipeline is Markovian)
        self.extended_state = {'C' : np.ndarray,'F' : np.ndarray,'T' : np.ndarray} # 'C' (candidate keypoints), 'F' (first observations of the tracked keypoint), 
                                                                                   # 'T' (camera pose at the first observation of the keypoint)


class VOInitializer():
    
    '''
    Provides the functionality to initialize to Visual Odometry Pipeline
    1) estimating the pose of frame2 -> T
        - (func) getKeypointMatches -> keypoints
        - (func) getPoseEstimate -> T
        
    2) estimating the 3D landmarks -> S (state)
        - (func) get_3D_landmarks -> S
    
    (we aim to make this estimation as accurate as possible)

    Inputs: 
        frame1: HxW np.ndarray
        frame2: HxW np.ndarray

    Outputs:
        T: 4x4 np.ndarray representing pose (T = [R|t]])
        S: dictionary with keypoints 'P' (keys) and 3D landmarks 'X' (values)
    '''
    
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K

    def getKeypointMatches(self, frame1: np.ndarray, frame2: np.ndarray) -> (np.array, np.array):
        ''' 
        Match keypoints between two frames using different detectors (default = SIFT) and the SIFT descriptor
        
        Args:
            frame1: HxW np.ndarray
            frame2: HxW np.ndarray
        
        Returns:
            good_kps_f1: array of keypoints in frame1
            good_kps_f2: array of keypoints in frame2
        '''

        if config['init_detector'] =='shi':
            
            # Initialize feauture detector and descriptor
            # Shi-Tomashi corner detector, to test and figure out (will probably have to change the type of kps_f1 to cv.KeyPoint)
            kps_f1 = cv2.goodFeaturesToTrack(frame1, maxCorners=600, qualityLevel=0.03, minDistance=10, mask=None, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)
            kps_f2 = cv2.goodFeaturesToTrack(frame2, maxCorners=600, qualityLevel=0.03, minDistance=10, mask=None, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)
            
            # Convert keypoints to cv2.KeyPoint format
            kps_f1 = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in kps_f1]
            kps_f2 = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in kps_f2]
            
        if config['init_detector'] == 'sift':

            sift = cv2.SIFT.create()
            kps_f1 = sift.detect(frame1, None)
            kps_f2 = sift.detect(frame2, None)
            
        if config['init_descriptor'] == 'sift':
        
            # SIFT corner descriptor
            sift = cv2.SIFT.create()
            _, des1 = sift.compute(frame1, kps_f1) # describes the features in frame 1
            _, des2 = sift.compute(frame2, kps_f2) # describes the features in frame 2
    
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

    def getPoseEstimate(self, kps_f1: list, kps_f2: list) -> np.ndarray:
        '''
        Estimatea the pose of the second frame based on the 2D-2D correspondences between the two frames (5-point RANSAC algorithm)
        The pose is always relative to the very first frame (which is the world frame)
        
        Args: 
            kps_f1: list of keypoints in frame1
            kps_f2: list of keypoints in frame2
            
        Returns: 
            T: 3x4 np.ndarray (T = [R|t]) (transformation matrix) representing the pose of frame2 with respect to frame1 
        '''
        
        # Use the 5-point RANSAC to compute the essential matrix, taking outliers into account (5-point algorithm uses the epipolar geometry to compute the essential matrix)
        # But, the 5-point algorithms can return up to 10 solution of the essential matrix

        # Compute the essential matrix using the RANSAC 5-point algorithms
        E, mask = cv2.findEssentialMat(kps_f1, kps_f2, self.K, cv2.RANSAC, 0.999, 1.0, None)

        # Given the essential matrix, 4 possible combinations of R and T are possile, but only one is in front of the camera (cheirality constraint)
        _, R, t, _ = cv2.recoverPose(E, kps_f1, kps_f2, self.K, mask=mask) #recoverPose enforces the 

        # Output transformation matrix (not homogeneous!)
        return np.hstack((R, t)), mask
    
    def get_2D_3D_landmarks_association(self, kps_1: np.ndarray, kps_2: np.ndarray, T_img1_img2: np.ndarray) -> Dict:
        
        # triangulate points
        m1 = self.K @ np.eye(3, 4)
        m2 = self.K @ T_img1_img2

        # set the initial state of the VO pipeline 
        state: Dict[tuple[np.ndarray, np.ndarray], np.ndarray] = {}
        XH = cv2.triangulatePoints(m1, m2, kps_1.T, kps_2.T).T #triagulated points are stored in homogeneous coordinates
        # state = {tuple(map(tuple, key)): value[:3]/value[3] for key, value in zip(zip(kps_1, kps_2), XH)} #add 3D points to the state, convert to euclidean coordinates, converts kps into tuples
        state['X'] = (XH[:,0:3].T / XH[:,3].T).T
        state['P'] = kps_1      
        
        return state  

class KeypointsToLandmarksAssociator():
    def __init__(self, K, current_pose):
        '''
        Initialize class
        '''
        self.K = K
        self.current_pose = current_pose


    def associateKeypointsToLandmarks(self, old_frame: np.ndarray, new_frame: np.ndarray, state: dict) -> dict:
        '''
        Associates keypoints from old image to features of the new image with the KLT algorithm

        Inputs:
            old_frame: HxW np.ndarray
            new_frame: HxW np.ndarray
            state: dict with keys 'P' (keypoints) and 'X' (3D landmarks)

        Outputs:
            associations: dictinary with keys 'P' and 'i'. associations['P'] contains 2D points from new_frame associated with previous landmarks
                          and associations['i'] contains list of indices to which the points are associated
        
        function to use cv2.calcOpticalFlowPyrLK()
        input: 
              prevImage
              nextImage
              prevPts -> vector of 2d points to track
              
        Output:
              nextPts -> vector of 2D points containing the calculated new 
                         positions of input features in the second image
              status -> vector with 1 if corresponding feature has been found, 0 if not
              error -> output vector of errors
        '''

        state['P'] = state['P'].astype(np.float32)
        next_points, status, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, state['P'], None)
        
        filter_well_tracked = np.hstack(status).astype(np.bool_)
        keypoints_well_tracked = next_points[filter_well_tracked]
        
        filter_tracking_lost = np.logical_not(filter_well_tracked)
        points_tracking_lost = next_points[filter_tracking_lost]
        
        # define the new state
        landmarks_corresponding = state['X'][filter_well_tracked]
        new_state = {'P': keypoints_well_tracked, 'X': landmarks_corresponding}
        
        return (new_state, points_tracking_lost)
    

class PoseEstimator():
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K

        """ Constants """
        self.REPOJ_THRESH = 2    # was 2, threshold on the reprojection error of points accepted as inliers
        self.CONFIDENCE = 0.99999   # desired confidence of result
        if config['find_new_candidates_method'] == 'sift-sift':
            self.CONFIDENCE = config['sift_ransac_confidence']
        if config['find_new_candidates_method'] == 'shi-mask':
            self.CONFIDENCE = config['shi_ransac_confidence']
    
    def estimatePose(self, associations: Dict[np.ndarray,np.ndarray], img_idx) -> np.ndarray:
        '''
        Computes the current pose from the associations found in previous steps

        Inputs:
            associations: dictionary with keys 'P' and 'X_old' that contain 2D points from the new frame and the corresponding matching in the state vector

        Outputs:
            T: 4x4 np.ndarray representing the pose of the new frame with respect to the world frame
        '''

        P = associations['P']
        X = associations['X']
        success, R_vec, t, inliers = cv2.solvePnPRansac(X,            # 3D points
                                                        P,            # 2D points               
                                                        self.K,       # intrinsic parameters
                                                        np.zeros(4),  # unknown parameter
                                                        flags=cv2.SOLVEPNP_ITERATIVE,
                                                        confidence=self.CONFIDENCE ,
                                                        reprojectionError=self.REPOJ_THRESH)
        
        R, _ = cv2.Rodrigues(R_vec)
        print("real angle ", np.arccos(R[0,0]) * 180 / np.pi)
        # add nonlinear refinement with --> solvePnPRefineLM
        
        if inliers is None:
            inliers = np.array([])
        print('len inliers', len(inliers))
        
        if len(inliers) < 4:
            print('less than 4 inliers (or None)! len(inliers) = ', len(inliers))
            
            # there need to be more keypoints to estimate the pose accurately
            # add brand new landmarks and keypoints to the state -> initialization
            
            # instantiate the VOInitializer
            VOInit = VOInitializer(self.K)

            # load current frame
            # load frame 4 steps in the future
            if config['dataset'] == 'kitti':
                kitti_path = 'kitti-dataset'  # replace with your path
                cur_frame = cv2.imread(f'{kitti_path}/05/image_0/{img_idx:06d}.png', cv2.IMREAD_GRAYSCALE)
                next_bootstrap_frame = cv2.imread(f'{kitti_path}/05/image_0/{img_idx+4:06d}.png', cv2.IMREAD_GRAYSCALE)
                
            elif config['dataset'] == 'malaga':
                malaga_path = 'malaga-urban-dataset-extract-07'  # replace with your path
                left_images = [img for img in os.listdir(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images') if img.endswith('left.jpg')]
                left_images.sort()
                cur_frame = cv2.imread(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{left_images[img_idx]}', cv2.IMREAD_GRAYSCALE)
                next_bootstrap_frame = cv2.imread(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{left_images[img_idx+4]}', cv2.IMREAD_GRAYSCALE)
                
            elif config['dataset'] == 'parking':
                parking_path = 'parking'  # replace with your path
                cur_frame = cv2.imread(f'{parking_path}/images/img_{img_idx:05d}.png', cv2.IMREAD_GRAYSCALE)
                next_bootstrap_frame = cv2.imread(f'{parking_path}/images/img_{img_idx+4:05d}.png', cv2.IMREAD_GRAYSCALE)

            else:
                raise ValueError("Invalid dataset selection")
            
            # detect, describe and match features
            kps_1, kps_2 = VOInit.getKeypointMatches(cur_frame, next_bootstrap_frame)
            print("len kps1", kps_1.shape)
            print("len kps2", kps_2.shape)

            # estimate pose
            img1_img2_pose_tranform, mask = VOInit.getPoseEstimate(kps_1, kps_2)
            mask = np.hstack(mask).astype(np.bool_)

            # triangulate landmarks, img1 and img2 are assumed to be far enough apart for accurate triangulation
            kps_1 = kps_1[mask, :]
            kps_2 = kps_2[mask,:]
            state = VOInit.get_2D_3D_landmarks_association(kps_1, kps_2, img1_img2_pose_tranform)
            
            # add the landmarks (state['X']) to the associations
            associations['X'] = np.vstack((associations['X'], state['X']))
            
            # add the keypoints of the current frame (kps_1) to the associations
            associations['P'] = np.vstack((associations['P'], kps_1))
            return self.estimatePose(associations, img_idx)
            

        inliers = np.hstack(inliers)
        print(inliers[0:10])
        print("shape pre modifia ", associations['X'].shape)
        associations['P'] = associations['P'][inliers,:]
        associations['X'] = associations['X'][inliers,:]
        print("shape pos modifia ", associations['X'].shape)
        T = np.concatenate([np.concatenate([R,t], axis=-1),np.array([[0,0,0,1]])], axis=0)
    
        return T

class LandmarkTriangulator():
    '''
    The LandmarkTriangulator class is designed to detect new keypoints, track the keypoints over multiple frames and add them to the state dictionary when certain conditions are met.
    
    For the detection of new keypoints, 3 alternative methods are implemented (configurable in the config.yaml file):
    1) def find_new_candidates_shi
    2) def find_new_candidates_sift_mask
    3) def find_new_candidates_sift_sift

    The def triangulate_landmark method is used to triangulate the new keypoints and add them to the state dictionary (when they meet certain conditions).

    '''
                                          
    def __init__(self, K, old_des):
        '''
        Initialize class
        '''
        self.K = K
        self.old_des = old_des

    def find_new_candidates_shi(self, new_frame: np.ndarray, state: dict, keypoints_well_tracked) -> list: 
        '''
        Detects new candidate keypoints in a given frame using Shi-Tomashi corner detector. 
        A mask is used to exclude already well-tracked keypoints to focus on new features.
            
        Inputs:
            new_frame: HxW np.ndarray representing the new frame for detection.
            state: dict containing current state information.
            keypoints_well_tracked: list of keypoints that are already being tracked (correctly).

        Outputs:
            non_duplicate_new_keypoints: list of newly detected keypoints that are not duplicates with the well tracked keypoints.
        '''
            
        # make a mask (type  type uint8/logical), exclude the well_tracked pixels from the Shi-Tomashi corner detector
        mask = np.ndarray(shape=(new_frame.shape[0], new_frame.shape[1]), dtype=np.uint8)
        # turn all the mask values to 255 by default
        mask.fill(255)
        # turn the mask to zero at the pixel positions of keypoints_well_tracked
        for i in range(len(keypoints_well_tracked)):
            y, x = int(keypoints_well_tracked[i][1]), int(keypoints_well_tracked[i][0])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                mask[y, x] = 0
        
        #Shi-Tomashi corner detector, to test and figure out (will probably have to change the type of kps_f1 to cv.KeyPoint)
        new_keypoints = cv2.goodFeaturesToTrack(new_frame, maxCorners=500, qualityLevel=0.03, minDistance=10, mask=mask, blockSize=4, gradientSize=3, useHarrisDetector=False, k=0.04)
        
        #Convert keypoints to cv2.KeyPoint format
        new_keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in new_keypoints]
        new_keypoints = np.array([kp.pt for kp in new_keypoints])
        
        non_duplicate_new_keypoints = new_keypoints

        return non_duplicate_new_keypoints

    def find_new_candidates_sift_mask(self, new_frame: np.ndarray, state: dict, keypoints_well_tracked) -> list: 
        '''
        Detects new candidate keypoints in a given frame using the SIFT corner detector. 
        A mask is used to exclude already well-tracked keypoints to focus on new features.
            
        Inputs:
            new_frame: HxW np.ndarray representing the new frame for detection.
            state: dict containing current state information.
            keypoints_well_tracked: list of keypoints that are already being tracked (correctly).

        Outputs:
            non_duplicate_new_keypoints: list of newly detected keypoints that are not duplicates with the well tracked keypoints.
        '''
        
        # make a mask (type  type uint8/logical), exclude the well_tracked pixels from the Shi-Tomashi corner detector
        mask = np.ndarray(shape=(new_frame.shape[0], new_frame.shape[1]), dtype=np.uint8)
        # turn all the mask values to 255 by default
        mask.fill(255)
        # turn the mask to zero at the pixel positions of keypoints_well_tracked
        for i in range(len(keypoints_well_tracked)):
            y, x = int(keypoints_well_tracked[i][1]), int(keypoints_well_tracked[i][0])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                mask[y, x] = 0
        
        # Use the sift descriptor to describe the new keypoints
        sift = cv2.SIFT.create()
        
        new_keypoints = sift.detect(new_frame, mask=mask) # describes the features in frame 1
        #new_keypoints, cur_des = sift.detectAndCompute(new_frame, mask=mask) # describes the features in frame 1

        new_keypoints = np.array([kp.pt for kp in new_keypoints])
        non_duplicate_new_keypoints = new_keypoints

        return non_duplicate_new_keypoints
    
    def find_new_candidates_sift_sift(self, new_frame: np.ndarray, state: dict, extended_state: dict) -> list:
        '''
        Detects new candidate keypoints in a given frame using the SIFT corner detector. 
        The corner is described with the SIFT corner descriptor and matched against the already well-tracked keypoints. 
        Good matches are removed to focus on new features.
            
        Inputs:
            new_frame: HxW np.ndarray representing the new frame for detection.
            state: dict containing current state information.
            keypoints_well_tracked: list of keypoints that are already being tracked (correctly).

        Outputs:
            non_duplicate_new_keypoints: list of newly detected keypoints that are not duplicates with the well tracked keypoints.
        '''
        
        # detect new keypoints in the current frame (the one where to find new candidates)
        sift = cv2.SIFT.create()
        keypoints, cur_des = sift.detectAndCompute(new_frame, None) 

        bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=False) # crossCheck=True is an alternative to the ratiotest (proposed by D. Lowe in SIFT paper)	
        kp_matches = bf.knnMatch(self.old_des, cur_des, k=2) # k=2: return the two best matches for each descriptor

        # Apply ratio test (preventing false matches)
        good_kp_matches = [m for m, n in kp_matches if m.distance < 0.8 * n.distance]

        # new_keypoints -> the good matches that didn't appear in the previous (old) image
        old_keypoints = np.array([keypoints[match.trainIdx].pt for match in good_kp_matches])
        keypoints = np.array([kp.pt for kp in keypoints])
        mask = np.isin(keypoints, old_keypoints, invert=True).all(axis=1)
        new_keypoints = keypoints[mask]
        
        # if debug:
        #     print("\n---- FIND NEW CANDIDATES ----")
        #     print(f"keypoints: {keypoints}")
        #     print(f"old_keypoints: {old_keypoints}")
        #     print(f"len(state['P']): {len(state['P'])}")
        #     print(f"len(keypoints): {len(keypoints)}")
        #     print(f"mask: {mask}")
        #     print(f"new_kp: {new_keypoints}")

        return (new_keypoints, cur_des)
        

    def triangulate_landmark(self, old_frame: np.ndarray, new_frame: np.ndarray, state: dict, extended_state: dict, new_pose) -> dict:
        '''
        The candidate keypoints (when meeting certain conditions) are triangulated and added to the state dictionary as landmarks.
        
        Inputs:
            frame: HxW np.ndarray
            extended_state: dict as defined in the main class
            
            features: 2xN np.ndarray containing 2D points detected in the last frame

        Output:
            new_landmarks: dictionary with keys 'P' and 'X'  for the new identified landmarks
        '''

        debug = False
        debug2 = False
        # print("\n\n---------- TRIANGULATE LANDMARK ----------")

        # 1) Track and update the existing candidate keypoints (from frame to frame)
        candidate_keypoints_1 = extended_state['C'].astype(np.float32)
        len_candidate_keypoints_1 = len(extended_state['C'])
        len_stateP_start = len(state['P'])
        if debug:
            print(f"candidate_keypoints_1: {candidate_keypoints_1}")
            print(f"len(candidate_keypoints_1): {len(candidate_keypoints_1)}")
            print(f"len(extended_state['C']) iniziale: {len(extended_state['C'])}")
            print(f"len(extended_state['F']) iniziale: {len(extended_state['F'])}")
            print(f"len(extended_state['T']) iniziale: {len(extended_state['T'])}")

        if len(candidate_keypoints_1)>0:

            candidate_keypoints_2, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, candidate_keypoints_1, None)
            # candidate_keypoints_2 is in floating values and depict the pixel coordinates of the next image (the can fall outside of the normal image boundaries)
            st = np.hstack(st).astype(np.bool_)
            if candidate_keypoints_2 is not None:
                candidate_keypoints_2_well_tracked = candidate_keypoints_2[st==1]
                candidate_keypoints_1_well_tracked = candidate_keypoints_1[st==1]
                candidate_keypoints_1_tracking_lost = candidate_keypoints_1[st==0]
                
            if len(candidate_keypoints_1_well_tracked) != len(candidate_keypoints_2_well_tracked):
                print(f"ERROR: candidate_keypoints_1_well_tracked and candidate_keypoints_2_well_tracked do not have the same length. len(candidate_keypoints_1_well_tracked)={len(candidate_keypoints_1_well_tracked)}!={len(candidate_keypoints_2_well_tracked)}=len(candidate_keypoints_2_well_tracked)")
                print("stop!!!")
                
            if debug:
                print("\n---- TRACK CANDIDATE KEYPOINTS ----")
                print(f"candidate_keypoints_2: {candidate_keypoints_2}")
                print(f"candidate_keypoints_2_well_tracked: {candidate_keypoints_2_well_tracked}")
                print(f"candidate_keypoints_1_well_tracked: {candidate_keypoints_1_well_tracked}")

            # Create a boolean mask to identify the positions of the tracked candidate keypoints to be updated
            if config['find_new_candidates_method'] == 'shi-mask':
                mask = np.isin(candidate_keypoints_2, candidate_keypoints_2_well_tracked).all(axis=1)
            if config['find_new_candidates_method'] == 'sift-sift':
                mask = np.isin(extended_state['C'], candidate_keypoints_1_well_tracked).all(axis=1)
            extended_state['C'][mask] = candidate_keypoints_2_well_tracked # Update all the candidate keypoints that have been retracked

            len_ckC_step3 = len(extended_state['C'])
            if debug:
                print("\n---- UPDATE C OF CANDIDATE POINTS THAT HAVE BEEN RETRACKED ----")
                print(f"mask: {mask}")
                print(f"extended_state['C'][mask]: {extended_state['C'][mask]}")
                print(f"extended_state['C']: {extended_state['C']}")
                print(f"len(extended_state['C']) dopo update retracked: {len(extended_state['C'])}")
                print(f"len(extended_state['F']) dopo update retracked: {len(extended_state['F'])}")
                print(f"len(extended_state['T']) dopo update retracked: {len(extended_state['T'])}")

            # Delete candidate keypoints that failed to track
            mask = np.isin(extended_state['C'], candidate_keypoints_1_tracking_lost).all(axis=1)
            # Get the indices of elements to be removed
            indices_to_remove = np.where(mask)[0]
            # Check if any elements are found before attempting removal
            if len(indices_to_remove) > 0:
                # Proceed to remove them
                extended_state['C'] = np.delete(extended_state['C'], indices_to_remove, axis=0)
                extended_state['F'] = np.delete(extended_state['F'], indices_to_remove, axis=0)
                extended_state['T'] = np.delete(extended_state['T'], indices_to_remove, axis=0)

            len_candidate_keypoints_2 = len(extended_state['C'])
            if debug:
                print("\n---- REMOVE CANDIDATE KEYPOINTS THAT HAVE NOT BEEN RETRACKED ----")
                print(f"candidate_keypoints_1_tracking_lost: {candidate_keypoints_1_tracking_lost}")
                print(f"mask: {mask}")
                print(f"indices_to_remove: {indices_to_remove}")
                print(f"extended_state['C']: {extended_state['C']}")
                print(f"extended_state['F']: {extended_state['F']}")
                print(f"extended_state['T']: {extended_state['T']}")
                print(f"len(extended_state['C']) iniziale: {len_candidate_keypoints_1}")
                print(f"len(extended_state['C']) dopo rimozione untracked: {len(extended_state['C'])}")
                print(f"len(extended_state['F']) dopo rimozione untracked: {len(extended_state['F'])}")
                print(f"len(extended_state['T']) dopo rimozione untracked: {len(extended_state['T'])}")
                
        if len(candidate_keypoints_1)==0:
            candidate_keypoints_1_well_tracked = []

        # 2) Add new candidate keypoints
        new_candidates_list = []
        cur_des = []
        if config['find_new_candidates_method'] == 'shi-mask':
            new_candidates_list = self.find_new_candidates_shi(new_frame, state, candidate_keypoints_1_well_tracked)
        elif config['find_new_candidates_method'] == 'sift-mask':
            new_candidates_list = self.find_new_candidates_sift_mask(new_frame, state, candidate_keypoints_1_well_tracked)
        elif config['find_new_candidates_method'] == 'sift-sift':
            new_candidates_list, cur_des = self.find_new_candidates_sift_sift(new_frame, state, extended_state)
            
        
        if debug:
            print(f"new candidates list: {new_candidates_list}")
            
        if len(new_candidates_list) > 0:
            new_candidates = {}
            new_candidates['C'] = new_candidates_list
            new_candidates['F'] = new_candidates_list
            new_candidates['T'] = np.stack([(np.eye(4) @ new_pose) for _ in range(len(new_candidates_list))])

            if len(extended_state['C']) == 0:
                extended_state['C'] = new_candidates['C']
                extended_state['F'] = new_candidates['F']
                extended_state['T'] = new_candidates['T']
            else:
                extended_state['C'] = np.concatenate((extended_state['C'], new_candidates['C']), axis=0)
                extended_state['F'] = np.concatenate((extended_state['F'], new_candidates['F']), axis=0)
                extended_state['T'] = np.concatenate((extended_state['T'], new_candidates['T']), axis=0) 

            len_ckC_step4 = len(extended_state['C'])
            if debug:
                print("\n---- ADD NEW CANDIDATE KEYPOINTS THAT HAVE NEWLY BEEN TRACKED ----")
                print(f"new_candidates['C']: {new_candidates['C']}")
                print(f"new_candidates['F']: {new_candidates['F']}")
                print(f"new_candidates['T']: {new_candidates['T']}")
                print(f"extended_state['C']: {extended_state['C']}")
                print(f"extended_state['F']: {extended_state['F']}")
                print(f"extended_state['T']: {extended_state['T']}")
                print(f"len(extended_state['C']) dopo add new: {len(extended_state['C'])}")
                print(f"len(extended_state['F']) dopo add new: {len(extended_state['F'])}")
                print(f"len(extended_state['T']) dopo add new: {len(extended_state['T'])}")


        # 3) validate new points (when the baseline is big enough to produce accurate triangulation) (from candidate keypoints to real keypoints used for triangulation)
        treshold = 1
        if config['find_new_candidates_method'] == 'sift-sift':
            treshold = config['sift_alpha']
        if config['find_new_candidates_method'] == 'shi-mask':
            treshold = config['shi_alpha']
        
        #debug
        alphas = []
        indices_to_validate = []
        if debug:
            print("\n---- VALIDATE NEW POINTS ----")
            print(f"len(extended_state['C']): {len(extended_state['C'])}")
        # Consider all candidate keypoints
        for idx in range(len(extended_state['C'])):
            # Compute the bearing vector corresponding to the current observation
            uc, vc = extended_state['C'][idx]
            wc = np.sqrt(uc**2 + vc**2 + 1)
            vector_a = np.array([uc/wc, vc/wc, 1/wc])

            # Compute the bearing vector corresponding to the first observation
            uf, vf = extended_state['F'][idx]
            wf = np.sqrt(uf**2 + vf**2 + 1)
            vector_b = np.array([uf/wf, vf/wf, 1/wf])

            # Compute the angle between the two bearing vectors
            cos = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
            # Ensure the value is within the valid range for arccosine
            cos = np.clip(cos, -1.0, 1.0)
            # Calculate the angle
            alpha = np.degrees(np.arccos(cos))
            alphas.append(alpha)

            # Confront the angle with a treshold: if bigger, proceed to validate
            if alpha > treshold:
                indices_to_validate.append(idx)
            if debug:
                # print(f"extended_state['C'][idx]: {extended_state['C'][idx]}")
                # print(f"vector_a: {vector_a}")
                # print(f"vector_b: {vector_b}")
                # print(f"cos: {cos}")
                # print(f"alpha: {alpha}")
                if alpha > treshold:
                    print(f"alpha > treshold: {alpha > treshold}")
                # print(f"indices_to_validate: {indices_to_validate}")

        if debug: 
            print(f"ALL indices_to_validate: {indices_to_validate}")
            print(f"number of validated ck: {len(indices_to_validate)}")
        point_3d_prin = []
        pose_3d_prin = []
        for idx in indices_to_validate:
            # Add homogeneous coordinates (1) to the 2D points
            # point1 = np.hstack((extended_state['C'][idx], np.array((1))))
            # point2 = np.hstack((extended_state['F'][idx], np.array((1))))
            # print(f"point1: {point1}")
            # print(f"point2: {point2}")
            pose_3d_prin.append(extended_state['T'][idx])
            # Triangulate the validated keypoint
            point_3d_hom = cv2.triangulatePoints(self.K @ extended_state['T'][idx][0:3,:], self.K @ new_pose[0:3,:], extended_state['F'][idx].T,  extended_state['C'][idx].T)
            # Convert homogeneous coordinates to 3D coordinates
            #point_3d = cv2.convertPointsFromHomogeneous(point_3d_hom.T).reshape(-1, 3)
            point_3d = (point_3d_hom[0:3].T / point_3d_hom[3].T).T

            # Add the validated keypoint to the state
            state['P'] = np.concatenate((state['P'], extended_state['C'][idx].reshape(1,2)), axis=0)
            state['X'] = np.concatenate((state['X'], point_3d.reshape(1,3)), axis=0)
            point_3d_prin.append(point_3d.reshape(3))
            
            if debug:
                # print(f"point1: {point1}")
                # print(f"point2: {point2}")
                print(f"point_3d_hom: {point_3d_hom}")
                print(f"point_3d: {point_3d}")
                print(f"extended_state['C'] validated: {extended_state['C'][idx]}")
                # print(f"state['P'] updated: {state['P']}")
                # print(f"state['X'] updated: {state['X']}")
        point_3d_prin = np.array(point_3d_prin)
        # print("poin 3d prin")
        # print(point_3d_prin.shape)
        # print("drawing ......")
        # plt.imshow(new_frame)
        # plt.scatter(extended_state['C'][indices_to_validate,0],extended_state['C'][indices_to_validate,1], color='blue', marker='o', label='Points')
        # plt.xlim((0,1200))
        # plt.plot()
        # plt.show()

        # if point_3d_prin.shape[0] > 0:
        #     axis = np.linalg.inv(new_pose) @ np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), np.ones((4,1)).T))
        #     plt.plot([axis[0,3],axis[0,0]],[axis[2,3], axis[2,0]], 'b-')
        #     plt.plot([axis[0,3],axis[0,2]],[axis[2,3], axis[2,2]], 'r-')
        #     for pose in pose_3d_prin:
        #         ax = np.linalg.inv(pose) @ np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), np.ones((4,1)).T))
        #         plt.plot([ax[0,3],ax[0,0]],[ax[2,3], ax[2,0]], 'b-')
        #         plt.plot([ax[0,3],ax[0,2]],[ax[2,3], ax[2,2]], 'g-')
        #     plt.scatter(point_3d_prin[:,0],point_3d_prin[:,2], color='blue', marker='o')
        #     plt.xlabel('X-axis')
        #     plt.ylabel('Z-axis')
        #     plt.title('augusto Visualization')
        #     plt.legend() # Show legend
        #     plt.show() # Show the plot
        # if len(alphas) > 0:
        #     print("alphas")
        #     alphas_array = np.array(alphas)
        #     print("alphas mean ",np.mean(alphas_array))
        #     print("median ", np.median(alphas_array))
        #     print("max value ", np.max(alphas_array))
        #     print("------ ")
            

        # REMOVE THE VALIDATED KEYPOINTS FROM THE CANDIDATE LIST
        if len(indices_to_validate) > 0:
            extended_state['C'] = np.delete(extended_state['C'], indices_to_validate, axis=0)
            extended_state['F'] = np.delete(extended_state['F'], indices_to_validate, axis=0)
            extended_state['T'] = np.delete(extended_state['T'], indices_to_validate, axis=0)

        len_ckC_step5 = len(extended_state['C'])
        
        if debug:
            print("\n---- REMOVE THE VALIDATED KEYPOINTS FROM THE CANDIDATE LIST ----")
            print(f"extended_state['C'] updated: {extended_state['C']}")
            print(f"extended_state['F'] updated: {extended_state['F']}")
            print(f"extended_state['T'] updated: {extended_state['T']}")
            print(f"len(extended_state['C']) dopo removal validated: {len(extended_state['C'])}")
            print(f"len(extended_state['F']) dopo removal validated: {len(extended_state['F'])}")
            print(f"len(extended_state['T']) dopo removal validated: {len(extended_state['T'])}")

        if debug2:
            print("\n\n---- SUM-UP CANDIDATE KEYPOINTS----")
            print(f"len_candidate_keypoints_1 start: {len_candidate_keypoints_1}")
            if len(candidate_keypoints_1)>0:
                print(f"len_candidate_keypoints_2 after removal non-tracked: {len_candidate_keypoints_2}")
                print(f"len_ckC_step3 after update retracked: {len_ckC_step3}")
            if len(new_candidates_list) > 0:
                print(f"len_ckC_step4 after add new ck: {len_ckC_step4}")
            print(f"len_ckC_step5 after removal validated: {len_ckC_step5}")

            print("---- SUM-UP STATE----")
            print(f"len_stateP_start: {len_stateP_start}")
            print(f"len_stateP_now: {len(state['P'])}")



        return (state, extended_state, cur_des)