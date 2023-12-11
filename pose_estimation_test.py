
import numpy as np
from typing import Dict
import cv2

class PoseEstimator():
    def __init__(self, K):
        '''
        Initialize class
        '''
        self.K = K

        """ Constants """
        self.REPOJ_THRESH = 1      # threshold on the reprojection error of points accepted as inliers
        self.CONFIDENCE = 0.9999   # dsired confidence of result
    
    def estimatePose(self, associations: Dict[np.ndarray,np.ndarray]) -> np.ndarray:
        '''
        Computes the current pose from the associations found in previous steps

        Inputs:
            associations: dictionary with keys 'P' and 'X_old' that contain 2D points from the new frame and the corresponding matching in the state vector

        Outputs:
            T: 4x4 np.ndarray representing the pose of the new frame with respect to the world frame
        '''

        P = associations['P'].T
        X = associations['X'].T

        success, R_vec, t, inliers = cv2.solvePnPRansac(X,            # 3D points
                                                        P,            # 2D points               
                                                        self.K,       # instarinsic parameters
                                                        np.zeros(4),  # unknown parameter
                                                        flags=cv2.SOLVEPNP_ITERATIVE,
                                                        confidence=self.CONFIDENCE ,
                                                        reprojectionError=self.REPOJ_THRESH)
        
        R, _ = cv2.Rodrigues(R_vec)

        print(inliers)
        # add nonlinear refinement with --> solvePnPRefineLM
        
        T = np.concatenate([np.concatenate([R,t], axis=-1),np.array([[0,0,0,1]])], axis=0)
        return T
    

if __name__ == "__main__":

    """ Some constants for testing """
    SCALE = 10
    N_inliers = 20
    N_outliers = 5
    N = N_inliers + N_outliers
    K = np.array([[1, 0, 1],
                  [0, 1, 1],
                  [0, 0, 1]]).astype('float')
    
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    t = np.array([0, 0, 0]).T

    """ Creation of a 3D point cloud """
    points3D = np.random.rand(3,N) * SCALE

    print("points3D", points3D.shape)

    """ Projection of inliers that gives 2D correct matches """
    rotated3D = np.tensordot(R, points3D, axes=[1,0]) + np.repeat(np.reshape(t,(3,1)), N, axis=-1)
    points2D_omo = np.tensordot(K,rotated3D,axes=[1,0])
    points2D = points2D_omo[:-1,:] / points2D_omo[-1,:]
    print("points2D", points2D.shape)

    """ Corrupt the data in order to add outliers """
    pointCloud = rotated3D
    pointCloud[:,N_inliers:] = np.random.rand(3,N_outliers) * SCALE
    print("pointCloud", pointCloud.shape)

    """ Now that the last N_outliers points in pointCloud are not correlated to the corresponding 2D points of points2D we perform camera localization """
    posEst = PoseEstimator(K)

    T_est = posEst.estimatePose({'P': points2D, 'X': pointCloud})

    print("T_est",T_est)