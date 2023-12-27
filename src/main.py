# %% 

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from vo_pipeline import *

def cross2Matrix(x):
    """ Antisymmetric matrix corresponding to a 3-vector
     Computes the antisymmetric matrix M corresponding to a 3-vector x such
     that M*y = cross(x,y) for all 3-vectors y.

     Input: 
       - x np.ndarray(3,1) : vector

     Output: 
       - M np.ndarray(3,3) : antisymmetric matrix
    """
    M = np.array([[0,   -x[2], x[1]], 
                  [x[2],  0,  -x[0]],
                  [-x[1], x[0],  0]])
    return M

# Setup
if config['dataset'] == 'kitti':
    # Set kitti_path to the folder containing "05" and "poses"
    kitti_path = '../kitti'  # replace with your path
    assert os.path.exists(kitti_path), "KITTI path does not exist"
    ground_truth = np.loadtxt(f'{kitti_path}/poses/05.txt')[:, -9:-7]
    last_frame = 4540
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])
    
    bootstrap_frames = [0, 2]
    img0 = cv2.imread(f'{kitti_path}/05/image_0/{bootstrap_frames[0]:06d}.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f'{kitti_path}/05/image_0/{bootstrap_frames[1]:06d}.png', cv2.IMREAD_GRAYSCALE)

elif config['dataset'] == 'malaga':
    # Set malaga_path to the folder containing Malaga dataset
    malaga_path = 'path_to_malaga_dataset'  # replace with your path
    assert os.path.exists(malaga_path), "Malaga path does not exist"
    left_images = [img for img in os.listdir(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images') if img.endswith('.png')]
    left_images.sort()
    last_frame = len(left_images)
    K = np.array([[621.18428, 0, 404.0076],
                  [0, 621.18428, 309.05989],
                  [0, 0, 1]])
    
    bootstrap_frames = [0, 2]
    img0 = cv2.imread(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{left_images[bootstrap_frames[0]]}', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{left_images[bootstrap_frames[1]]}', cv2.IMREAD_GRAYSCALE)

elif config['dataset'] == 'parking':
    # Set parking_path to the folder containing parking dataset
    parking_path = 'parking'  # replace with your path
    assert os.path.exists(parking_path), "Parking path does not exist"
    last_frame = 598
    K = np.loadtxt(f'{parking_path}/K.txt')
    ground_truth = np.loadtxt(f'{parking_path}/poses.txt')[:, -9:-7]
    
    bootstrap_frames = [0, 2]
    img0 = cv2.imread(f'{parking_path}/images/img_{bootstrap_frames[0]:05d}.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f'{parking_path}/images/img_{bootstrap_frames[1]:05d}.png', cv2.IMREAD_GRAYSCALE)

else:
    raise ValueError("Invalid dataset selection")


### 1 - Initialization
#instantiate BestVision:
vision = BestVision(K)
# instantiate the VOInitializer
VOInit = VOInitializer(K)
# instantiate Landmark association
associate = KeypointsToLandmarksAssociator(K)

initial_good_kp_matches, kps1, kps2 = VOInit.get_keypoint_matches(img0, img1)
# print(len(initial_good_kp_matches))
# print(len(kps1))
# print(len(kps2))
# img3 = cv2.drawMatches(img0,kps1,img1,kps2,initial_good_kp_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)

T_hom = VOInit.estimate_pose(kps1, kps2)
t_inv = np.linalg.inv(T_hom)
axis = t_inv @ np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), np.ones((4,1)).T))
print(t_inv[0:3,:])
m1 = K @ t_inv[0:3,:]
starting_pose = np.hstack((np.eye(3), np.zeros((3,1))))
m0 = K @ starting_pose
Landmarks3D_H =cv2.triangulatePoints(m0, m1, kps1.T, kps2.T)
Landmarks3D = Landmarks3D_H[0:3,:]/Landmarks3D_H[3,:]
#TODO resolve problems on the generation of points

filter = np.linalg.norm(Landmarks3D, axis = 0) < 4
print("filter len ", filter.shape)
Landmarks3D_filtered = Landmarks3D[:, filter]
plt.scatter(Landmarks3D_filtered[0,:], Landmarks3D_filtered[2,:], color='blue', marker='o', label='Points')
# plt.scatter(P[0,:], P[2,:], color='red', marker='o', label='Points')
plt.plot([axis[0,3],axis[0,0]],[axis[2,3], axis[2,0]], 'r-')
plt.plot([axis[0,3],axis[0,2]],[axis[2,3], axis[2,2]], 'g-')
# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Z-axis')
plt.ylim((0,10))
plt.xlim((-5,5))
plt.title('2D Points Visualization')

# Show legend
plt.legend()

# Show the plot
plt.show()

plt.imshow(img1)
points = kps1[filter, :]
print("size filtered points ", points.shape)
plt.scatter(kps1[:,0], kps1[:,1], color='blue', marker='o', label='Points')
plt.scatter(points[:,0], points[:,1], color='red', marker='o', label='Points')
plt.plot()
plt.show()

plt.imshow(img1)
points2 = kps2[filter,:]
plt.scatter(kps2[:,0], kps2[:,1], color='blue', marker='o', label='Points')
plt.scatter(points2[:,0], points2[:,1], color='red', marker='o', label='Points')
plt.plot()
plt.show()

vision.update_state(kps2, Landmarks3D.T)

if ds == 0:
    img2 = cv2.imread(f'{kitti_path}/05/image_0/{5:06d}.png', cv2.IMREAD_GRAYSCALE)

elif ds == 1:
    img2 = cv2.imread(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{left_images[bootstrap_frames[2]]}', cv2.IMREAD_GRAYSCALE)

elif ds == 2:
    img2 = cv2.imread(f'{parking_path}/images/img_{bootstrap_frames[2]:05d}.png', cv2.IMREAD_GRAYSCALE)

else:
    raise ValueError("Invalid dataset selection")

state_2 = associate.associateKeypoints(img1,img2,vision.state)


# %%
