#%%

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vo_pipeline import *
from visual import Visual

debug = config['debug']
visualize = config['visualization']

# Setup
if config['dataset'] == 'kitti':
    # Set kitti_path to the folder containing "05" and "poses"
    kitti_path = 'kitti-dataset'  # replace with your path
    assert os.path.exists(kitti_path), "KITTI path does not exist"
    ground_truth = np.loadtxt(f'{kitti_path}/poses/05.txt')[:, -9:-7]
    kitti_images = [img for img in os.listdir(f'{kitti_path}/05/image_0') if img.endswith('.png')]
    last_frame = len(kitti_images) - 1 
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])
    
    bootstrap_frames = [0,6] # [0,6]
    img0 = cv2.imread(f'{kitti_path}/05/image_0/{bootstrap_frames[0]:06d}.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f'{kitti_path}/05/image_0/{bootstrap_frames[1]:06d}.png', cv2.IMREAD_GRAYSCALE)

elif config['dataset'] == 'malaga':
    # Set malaga_path to the folder containing Malaga dataset
    malaga_path = 'malaga-urban-dataset-extract-07'  # replace with your path
    assert os.path.exists(malaga_path), "Malaga path does not exist"
    #ground_truth = np.loadtxt(f'{malaga_path}/poses/05.txt')[:, -9:-7]
    left_images = [img for img in os.listdir(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images') if img.endswith('.jpg')]
    left_images.sort()
    last_frame = len(left_images) -1 
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
    parking_images = [img for img in os.listdir(f'{parking_path}/images') if img.endswith('.png')]
    last_frame = len(parking_images) - 1
    K = np.array([[331.37, 0, 320],
                [0, 369.568, 240],
                [0, 0, 1]])
    
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
print("len kps2", kps_2.shape)

# estimate pose
img1_img2_pose_tranform, mask = VOInit.getPoseEstimate(kps_1, kps_2)
mask = np.hstack(mask).astype(np.bool_)

# triangulate landmarks, img1 and img2 are assumed to be far enough apart for accurate triangulation
kps_1 = kps_1[mask, :]
kps_2 = kps_2[mask,:]
state = VOInit.get_2D_3D_landmarks_association(kps_1, kps_2, img1_img2_pose_tranform)

print("dimensione ", img1_img2_pose_tranform.shape)
T_hom = np.vstack((img1_img2_pose_tranform, np.array([0,0,0,1])))
t_inv = np.linalg.inv(T_hom)
axis = t_inv @ np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), np.ones((4,1)).T))

pos_h = np.zeros(4)
pos_h[3] = 1
pos = (np.linalg.inv(T_hom) @ pos_h)[0:3]

filter = np.linalg.norm(state['X'], axis = 1) < 50
filter_add = np.linalg.norm(state['X'], axis = 1) > 3
filter = filter * filter_add
print("filter len ", filter.shape)
print("X shape ", state['X'].shape)
X_filtered = state['X'][filter,:]
    
if visualize:
    # plot the initialization images
    plt.figure(figsize=(10, 3.5))
    plt.imshow(img0, cmap='gray')
    plt.scatter(kps_1[:, 0], kps_1[:, 1], c='r', s=20, label = 'keypoints')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.title('Initialization Image 1 (frame %d)' % bootstrap_frames[0])
    plt.legend()
    plt.savefig('initialization-plots/%s_initialization_img1_%s-%s_frames_detector_%s.png' % (config['dataset'], bootstrap_frames[0], bootstrap_frames[1], config['init_detector']))
    plt.show()

    plt.figure(figsize=(10, 3.5))
    plt.imshow(img1, cmap='gray')
    plt.scatter(kps_2[:, 0], kps_2[:, 1], c='r', s=20, label = 'keypoints')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.title('Initialization Image 2 (frame %d)' % bootstrap_frames[1])
    plt.legend()
    plt.savefig('initialization-plots/%s_initialization_img2_%s-%s_frames_detector_%s.png' % (config['dataset'], bootstrap_frames[0], bootstrap_frames[1], config['init_detector']))
    plt.show()

#     # 3D plot of the initialization 3D landmarks (X)
#     # fig = plt.figure(figsize=(10, 10))
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', s=20)
#     # ax.set_xlabel('x (m)')
#     # ax.set_ylabel('y (m)')
#     # ax.set_zlabel('z (m)')
#     # ax.set_title('3D landmarks (X)')
#     # plt.show()

    # triangulated 3D Points Visualization (z-axis) of the initialization
    plt.scatter(X_filtered[:,0], X_filtered[:,2], color='blue', marker='o', label='Points')
    # plt.plot([axis[0,3],axis[0,0]],[axis[2,3], axis[2,0]], 'r-')
    # plt.plot([axis[0,3],axis[0,2]],[axis[2,3], axis[2,2]], 'g-')
    # plot the origin
    plt.plot(0, 0, 'g.', label='Start Position')
    plt.plot(pos[0], pos[2], 'r.', label='Next Position')
    plt.xlabel('X-axis')
    plt.ylabel('Z-axis')
    plt.ylim((0,50))
    plt.xlim((-15,15))
    plt.title('Pose Estimate and triangulated 3D landmarks of the initialization')
    plt.legend() # Show legend
    plt.savefig('initialization-plots/%s_initialization_3Dlandmarks__%s-%s_frames_detector_%s.png' % (config['dataset'], bootstrap_frames[0], bootstrap_frames[1], config['init_detector']))
    plt.show() # Show the plot

    # plot all and filtered 2D keypoints (img 1)
    # points = kps_1[filter, :]
    # plt.imshow(img1)
    # print("size filtered points ", points.shape)
    # plt.scatter(kps_1[:,0], kps_1[:,1], color='blue', marker='o', label='All keypoints')
    # plt.scatter(points[:,0], points[:,1], color='red', marker='o', label='Filtered keypoints')
    # plt.title('filtered 2d keypoints image 1')
    # plt.plot()
    # plt.show()

    # plot all and filtered 2D keypoints (img 2)
    # points2 = kps_2[filter,:]
    # plt.imshow(img1)
    # plt.scatter(kps_2[:,0], kps_2[:,1], color='blue', marker='o', label='All keypoints')
    # plt.scatter(points2[:,0], points2[:,1], color='red', marker='o', label='Filtered keypoints')
    # plt.title('filtered 2d keypoints image 2')
    # plt.plot()
    # plt.show()


### - Continuous Operation
extended_state = {}
extended_state['C'] = np.array([])
extended_state['F'] = np.array([])
extended_state['T'] = np.array([])

sift = cv2.SIFT.create()
_, old_des = sift.detectAndCompute(img1, None) # this should come from the initialization and we should start from img2

visualization = False
visual = Visual(K)
#instantiate BestVision:
vision = BestVision(K) 
vision.state = state
vision.extended_state = extended_state
cur_pose = img1_img2_pose_tranform # cur_pose is the position of frame 2 now!
if debug:
    print(f"img1_img2_pose_tranform: {img1_img2_pose_tranform}")
    print(f"img1_img2_pose_tranform.shape: {img1_img2_pose_tranform.shape}")
    
associate = KeypointsToLandmarksAssociator(K, T_hom)
pose_estimator = PoseEstimator(K)


#axis_list = []
X_plotting = np.array([])
poses_plotting = np.array([])
final_frame = last_frame # if you want to adjust the end frame
for img_idx in range(bootstrap_frames[1],final_frame): #was 3, 700
    print(f"\n\n\n\n---------- IMG {img_idx} ----------")
    # loading the next image
    if config['dataset'] == 'kitti':
        img2 = cv2.imread(f'{kitti_path}/05/image_0/{img_idx:06d}.png', cv2.IMREAD_GRAYSCALE)

    elif config['dataset'] == 'malaga':
        img2 = cv2.imread(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{left_images[img_idx]}', cv2.IMREAD_GRAYSCALE)

    elif config['dataset'] == 'parking':
        img2 = cv2.imread(f'{parking_path}/images/img_{img_idx:05d}.png', cv2.IMREAD_GRAYSCALE)

    else:
        raise ValueError("Invalid dataset selection")

    # associate the new keypoints (img 2) to the existing landmarks (X) 
    new_state, keypoints_tracking_lost = associate.associateKeypointsToLandmarks(img1, img2, vision.state)
    if debug:
        print(f"keypoints_tracking_lost: {keypoints_tracking_lost}")
        print(f"len(state_2['P']): {len(new_state['P'])}")

    # estimate the pose of the new frame and update 
    new_pose = pose_estimator.estimatePose(new_state, img_idx) # T_world_newframe
    if debug:
        print(f"new_pose (T_world_newframe): {new_pose}")
    
    axis = np.linalg.inv(new_pose) @ np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), np.ones((4,1)).T))
    pos_h = np.zeros(4)
    pos_h[3] = 1
    pos = (np.linalg.inv(new_pose) @ pos_h)[0:3]
    
    # add the new pose to the plotting array
    poses_plotting = np.vstack((poses_plotting, pos)) if poses_plotting.size else pos
    
    # add the new 3D landmarks close <50 to the current position to the plotting aray
    X = vision.state['X']
    filter = np.linalg.norm(X - pos, axis = 1) < 50
    filter_add = np.linalg.norm(X - pos, axis = 1) > 3
    filter = filter * filter_add
    print("filter len ", filter.shape)
    print("X shape ", X.shape)
    X_filtered = X[filter,:]
    X_plotting = np.vstack((X_plotting, X_filtered)) if X_plotting.size else X_filtered
    
    # add the car path to the plotting array
    plt.xlabel('X-axis')
    plt.ylabel('Z-axis')
    # plt.axis('square')
    plt.title('Travelled Path and 3D Landmarks Visualization')
    
    # plot every 200 frames
    if img_idx % 200 == 0:
        # plot the 3D landmarks
        plt.scatter(X_plotting[:,0], X_plotting[:,2], color='blue', marker='o', label='3D Landmarks')

        # plot the ground truth path in green
        # plt.plot(ground_truth[:,0], ground_truth[:,1], 'g-', label='Ground Truth')

        # plot the travelled car path (positions) in red
        plt.plot(poses_plotting[:,0], poses_plotting[:,2], 'r-', label='Travelled Path')

        plt.legend() # Show legend
        plt.savefig('trajectory-plots/%s_trajectory__%s_%s-%s_frames.png' % (config['dataset'], config['find_new_candidates_method'], bootstrap_frames[0], img_idx))
        plt.clf()
        print('New plot saved!')

    # Add new landmark triangulations to the state
    landmark_triangulator = LandmarkTriangulator(K, old_des)
    if config['find_new_candidates_method'] == 'sift-sift-des-compare':
        new_state, extended_state, cur_des = landmark_triangulator.triangulate_landmark(img1, img2, new_state, extended_state, new_pose)
    else:
        new_state, extended_state, cur_des = landmark_triangulator.triangulate_landmark(img1, img2, new_state, extended_state, new_pose)
    
    visual.update(img2,new_state, new_pose)
    visual.render()
    
    # update the state
    vision.state = new_state
    vision.extended_state = extended_state
    img1 = img2
    cur_des = old_des
    
# plot the 3D landmarks
plt.scatter(X_plotting[:,0], X_plotting[:,2], color='blue', marker='o', label='3D Landmarks')

# plot the ground truth path in green
# plt.plot(ground_truth[:,0], ground_truth[:,1], 'g-', label='Ground Truth')

# plot the travelled car path (positions) in red
plt.plot(poses_plotting[:,0], poses_plotting[:,2], 'r-', label='Travelled Path')

plt.legend() # Show legend
plt.savefig('trajectory-plots/%s_trajectory__%s_%s-%s_frames.png' % (config['dataset'], config['find_new_candidates_method'], bootstrap_frames[0], img_idx))
plt.show()

# idxs = np.arange(0,2000,250)
# intervals = []
# for idx in idxs:
#     intervals.append((idx,idx+500))
# intervals.append((2250,2760))

# for interval in intervals:
#     plt.xlabel('X-axis')
#     plt.ylabel('Z-axis')
#     # plt.axis('square')
#     plt.title(f'Travelled Path and 3D Landmarks Visualization: from frame {interval[0]} to frame {interval[1]} KITTI')
#     # plot every 200 frames
#     # plot the 3D landmarks
#     plt.scatter(X_plotting[interval[0],interval[1]], X_plotting[interval[0],interval[1]], color='blue', marker='o', label='3D Landmarks')
#     # plot the ground truth path in green
#     # plt.plot(ground_truth[:,0], ground_truth[:,1], 'g-', label='Ground Truth')
#     # plot the travelled car path (positions) in red
#     plt.plot(poses_plotting[interval[0],interval[1]], poses_plotting[interval[0],interval[1]], 'r-', label='Travelled Path')
#     plt.legend() # Show legend
#     plt.savefig(f'KITTI DATASET: from frame {interval[0]} to frame {interval[1]}.png')
#     plt.clf()
# # plt.show()


#%%
