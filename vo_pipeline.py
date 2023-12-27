
import numpy as np
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

        self.K = K
        self.previous_image = np.ndarray
        self.state = {'P' : np.ndarray, 'X' : np.ndarray}
        self.candidate_keypoints = {'C' : np.ndarray,'F' : np.ndarray,'T' : np.ndarray}

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

        # new_candidates_list are the new keypoints identified by Ricky in the cur_frame and NOT present in the previous frames
        # => EXPLAIN TO RICKY HOW TO PASS THEM: I would say he can simply pass a list of these new_candidates
        # that he finds by taking the keypoints of cur_frame that have no correspondence in the prec_frame
        # ATTENTION: every time all the keyframes that have not yet been
        # validated are inserted in new_candidates, then I proceed to evaluate if they are actually new or if they had already been identified but NOT YET
        # validated

        # I proceed to evaluate which of these new_candidates had already been previously tracked and which are
        # new. I also proceed to remove the candidate_points that have not been re-identified



        # TRACK CANDIDATE KEYPOINTS
        p0 = candidate_keypoints['C']
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, cur_frame, p0, None)
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        if len(good_old) != len(good_new):
            print(f"ERRORE: good_old e good_new NON hanno la stessa lunghezza. len(good_old)={len(good_old)}!={len(good_new)}=len(good_new)")


        # REMOVE CANDIDATE KEYPOINTS THAT HAVE NOT BEEN RETRACKED
        bad_old = p0[st==0]
        # Create a boolean mask to identify the positions of candidate keypoints that have not been retracked
        mask = np.isin(candidate_keypoints['C'], bad_old)
        # Get the indices of elements to be removed
        indices_to_remove = np.where(mask)[0]
        # Check if any elements are found before attempting removal
        if indices_to_remove.size > 0:
            # Proceed to remove them
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
        # Create a boolean mask to identify the positions of the tracked candidate keypoints to be updated
        mask = np.isin(candidate_keypoints['C'], good_old)
        # Update all the candidate keypoints that have been retracked
        candidate_keypoints['C'][mask] = good_new
    

        # VALIDATE NEW POINTS
        treshold = 5
        indices_to_validate = []
        # Consider all candidate keypoints
        for idx in len(candidate_keypoints['C']):
            # Compute the bearing vector corresponding to the current observation
            uc, vc = candidate_keypoints['C'][idx]
            wc = np.sqrt(uc**2 + vc**2 + 1)
            vector_a = np.array([uc/wc, vc/wc, 1/wc])

            # Compute the bearing vector corresponding to the first observation
            uf, vf = candidate_keypoints['F'][idx]
            wf = np.sqrt(uf**2 + vf**2 + 1)
            vector_b = np.array([uf/wf, vf/wf, 1/wf])

            # Compute the angle between the two bearing vectors
            cos = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
            # Ensure the value is within the valid range for arccosine
            cos = np.clip(cos, -1.0, 1.0)
            # Calculate the angle
            alpha = np.degrees(np.arccos(cos))

            # Confront the angle with a treshold: if bigger, proceed to validate
            if alpha > treshold:
                indices_to_validate.append(idx)

        for idx in indices_to_validate:
            # Add homogeneous coordinates (1) to the 2D points
            point1 = np.hstack((candidate_keypoints['C'][idx], np.array((1))))
            point2 = np.hstack((candidate_keypoints['F'][idx], np.array((1))))
            
            # Triangulate the validated keypoint
            point_3d_hom = cv2.triangulatePoints(cur_pose, candidate_keypoints['T'][idx], point1.T, point2.T)
            # Convert homogeneous coordinates to 3D coordinates
            point_3d = cv2.convertPointsFromHomogeneous(point_3d_hom.T).reshape(-1, 3)

            # Add the validated keypoint to the state
            state['P'].append(candidate_keypoints['C'][idx])
            state['X'].append(point_3d)
            

        # Remove the validated keypoints from the candidate list
        candidate_keypoints['C'] = np.delete(candidate_keypoints['C'], indices_to_validate)
        candidate_keypoints['F'] = np.delete(candidate_keypoints['F'], indices_to_validate)
        candidate_keypoints['T'] = np.delete(candidate_keypoints['T'], indices_to_validate)


        

        pass

        