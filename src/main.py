# %%

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vo_pipeline import *

debug = config['debug']

# Setup
if config['dataset'] == 'kitti':
    # Set kitti_path to the folder containing "05" and "poses"
    kitti_path = 'kitti'  # replace with your path
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



# instantiate the VOInitializer
VOInit = VOInitializer(K)

### 1 - Initialization
# detect, describe and match features
kps_1, kps_2 = VOInit.getKeypointMatches(img0, img1)
print("len kps1", kps_1.shape)

# estimate pose
img1_img2_pose_tranform = VOInit.getPoseEstimate(kps_1, kps_2)

# triangulate landmarks
state = VOInit.get_2D_3D_landmarks_association(kps_1, kps_2, img1_img2_pose_tranform)
print(state.keys())
print("len P ",state['P'].shape)
X = state['X']
P = state['P']

# plot the initialization images
plt.figure(figsize=(10, 10))
plt.imshow(img0, cmap='gray')
plt.scatter(kps_1[:, 0], kps_1[:, 1], c='r', s=20)
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Image 1')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(img0, cmap='gray')
plt.scatter(kps_2[:, 0], kps_2[:, 1], c='r', s=20)
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Image 2')
plt.show()

# 3D plot of the initialization 3D landmarks (X)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', s=20)
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('z (m)')
# ax.set_title('3D landmarks (X)')
# plt.show()

# plot a filtered version of the 3D landmarks (X) (some bugs, comes from Riccardo)
print("dimensione ", img1_img2_pose_tranform.shape)
T_hom = np.vstack((img1_img2_pose_tranform, np.array([0,0,0,1])))
t_inv = np.linalg.inv(T_hom)
axis = t_inv @ np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), np.ones((4,1)).T))

filter = np.linalg.norm(X, axis = 1) < 50
filter_add = np.linalg.norm(X, axis = 1) > 3
filter = filter * filter_add
print("filter len ", filter.shape)
print("X shape ", X.shape)
X_filtered = X[filter,:]
plt.scatter(X_filtered[:,0], X_filtered[:,2], color='blue', marker='o', label='Points')
plt.plot([axis[0,3],axis[0,0]],[axis[2,3], axis[2,0]], 'r-')
plt.plot([axis[0,3],axis[0,2]],[axis[2,3], axis[2,2]], 'g-')
plt.xlabel('X-axis')
plt.ylabel('Z-axis')
plt.ylim((0,50))
plt.xlim((-15,15))
plt.title('2D Points Visualization')
plt.legend() # Show legend
plt.show() # Show the plot

# plot all and filtered 2D keypoints (img 1)
plt.imshow(img1)
points = kps_1[filter, :]
print("size filtered points ", points.shape)
plt.scatter(kps_1[:,0], kps_1[:,1], color='blue', marker='o', label='All keypoints')
plt.scatter(points[:,0], points[:,1], color='red', marker='o', label='Filtered keypoints')
plt.plot()
plt.show()

# plot all and filtered 2D keypoints (img 2)
plt.imshow(img1)
points2 = kps_2[filter,:]
plt.scatter(kps_2[:,0], kps_2[:,1], color='blue', marker='o', label='All keypoints')
plt.scatter(points2[:,0], points2[:,1], color='red', marker='o', label='Filtered keypoints')
plt.plot()
plt.show()


### - Continuous Operation
candidate_keypoints = {}
candidate_keypoints['C'] = np.array([])
candidate_keypoints['F'] = np.array([])
candidate_keypoints['T'] = np.array([])

sift = cv2.SIFT.create()
_, old_des = sift.detectAndCompute(img1, None)

cur_pose = img1_img2_pose_tranform
print(f"img1_img2_pose_tranform: {img1_img2_pose_tranform}")
print(f"img1_img2_pose_tranform.shape: {img1_img2_pose_tranform.shape}")

vision = BestVision(K) #instantiate BestVision:
vision.state = state
vision.candidate_keypoints = candidate_keypoints
associate = KeypointsToLandmarksAssociator(K, T_hom)
pose_estimator = PoseEstimator(K)

for img_idx in range(3,40):
    print(f"\n\n\n\n---------- IMG {img_idx} ----------")
    # loading the next image
    if config['dataset'] == 'kitti':
        img2 = cv2.imread(f'{kitti_path}/05/image_0/{img_idx:06d}.png', cv2.IMREAD_GRAYSCALE)

    elif config['dataset'] == 'malaga':
        img2 = cv2.imread(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{left_images[bootstrap_frames[2]]}', cv2.IMREAD_GRAYSCALE)

    elif config['dataset'] == 'parking':
        img2 = cv2.imread(f'{parking_path}/images/img_{bootstrap_frames[2]:05d}.png', cv2.IMREAD_GRAYSCALE)

    else:
        raise ValueError("Invalid dataset selection")

    # instantiate Landmark association

    state_2, new_candidates_list = associate.associateKeypoints(img1,img2, vision.state)
    if debug:
        print(f"new_candidates_list: {new_candidates_list}")
        print(f"len(state_2['P']): {len(state_2['P'])}")

    # estimate the pose of the new frame
    T_world_newframe = pose_estimator.estimatePose(state_2)
    if debug:
        print(f"T_world_newframe: {T_world_newframe}")
    associate.update_pose(T_world_newframe)

    # triangulate landmarks
    landmark_triangulator = LandmarkTriangulator(K, old_des)
    new_state, candidate_keypoints, cur_des = landmark_triangulator.triangulate_landmark(img1, img2, state_2, candidate_keypoints, new_candidates_list, T_world_newframe)
    
    # if debug:
    #     print(f"new_state: {new_state}")
    #     print(f"candidate_keypoints: {candidate_keypoints}")

    vision.state = new_state
    vision.candidate_keypoints = candidate_keypoints
    img1 = img2
    old_des = cur_des
    cur_pose = T_world_newframe
    # if debug:
    #     print(f"vision.state: {vision.state}")
    #     print(f"vision.candidate_keypoints: {vision.candidate_keypoints}")

# ***** DEBUG *****
# plot a filtered version of the 3D landmarks (X) (some bugs, comes from Riccardo)
print("dimensione ", img1_img2_pose_tranform.shape)
T_hom = np.vstack((img1_img2_pose_tranform, np.array([0,0,0,1])))
t_inv = np.linalg.inv(T_hom)
axis = t_inv @ np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), np.ones((4,1)).T))

t_inv_2 = np.linalg.inv(T_world_newframe)
axis_2 = t_inv_2 @ np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), np.ones((4,1)).T))

filter = np.linalg.norm(X, axis = 1) < 10
filter_add = np.linalg.norm(X, axis = 1) > 3
filter = filter * filter_add
print("filter len ", filter.shape)
print("X shape ", X.shape)
X_filtered = X[filter,:]
plt.scatter(X_filtered[:,0], X_filtered[:,2], color='blue', marker='o', label='Points')
plt.plot([axis[0,3],axis[0,0]],[axis[2,3], axis[2,0]], 'r-')
plt.plot([axis[0,3],axis[0,2]],[axis[2,3], axis[2,2]], 'r-')
plt.plot([axis_2[0,3],axis_2[0,0]],[axis_2[2,3], axis_2[2,0]], 'b-')
plt.plot([axis_2[0,3],axis_2[0,2]],[axis_2[2,3], axis_2[2,2]], 'b-')
plt.xlabel('X-axis')
plt.ylabel('Z-axis')
plt.ylim((0,10))
plt.xlim((-5,5))
plt.title('2D Points Visualization')
plt.legend() # Show legend
plt.show() # Show the plot
#*****************

#%%
