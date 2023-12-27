import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vo_pipeline import *

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

elif config['dataset'] == 'parking':
    # Set parking_path to the folder containing parking dataset
    parking_path = 'parking'  # replace with your path
    assert os.path.exists(parking_path), "Parking path does not exist"
    last_frame = 598
    K = np.loadtxt(f'{parking_path}/K.txt')
    ground_truth = np.loadtxt(f'{parking_path}/poses.txt')[:, -9:-7]

else:
    raise ValueError("Invalid dataset selection")

# Bootstrap
bootstrap_frames = [0, 4]  # replace with your bootstrap frame indices -> will be determined by a keyframe criteria

if config['dataset'] == 'kitti':
    img0 = cv2.imread(f'{kitti_path}/05/image_0/{bootstrap_frames[0]:06d}.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f'{kitti_path}/05/image_0/{bootstrap_frames[1]:06d}.png', cv2.IMREAD_GRAYSCALE)

elif config['dataset'] == 'malaga':
    img0 = cv2.imread(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{left_images[bootstrap_frames[0]]}', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f'{malaga_path}/malaga-urban-dataset-extract-07_rectified_800x600_Images/{left_images[bootstrap_frames[1]]}', cv2.IMREAD_GRAYSCALE)

elif config['dataset'] == 'parking':
    img0 = cv2.imread(f'{parking_path}/images/img_{bootstrap_frames[0]:05d}.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f'{parking_path}/images/img_{bootstrap_frames[1]:05d}.png', cv2.IMREAD_GRAYSCALE)

else:
    raise ValueError("Invalid dataset selection")


# instantiate the VOInitializer
VOInit = VOInitializer(K)

# detect, describe and match features
kps_1, kps_2 = VOInit.get_keypoint_matches(img0, img1)

# estimate pose
img1_img2_pose_tranform = VOInit.get_pose_estimate(kps_1, kps_2)

# triangulate points
m1 = K @ np.eye(3, 4)
m2 = K @ img1_img2_pose_tranform

# this implementation can be made faster (remove the for loop)
X = np.empty((len(kps_1), 3, 1))
for i in range(len(kps_1)):
    XH = cv2.triangulatePoints(m1, m2, kps_1[i], kps_2[i]) #triagulated points are stored in homogeneous coordinates
    X[i] = XH[:3] / XH[3] #convert to euclidean coordinates
    
# plot the initialization images
plt.figure(figsize=(10, 10))
plt.imshow(img0, cmap='gray')
plt.scatter(kps_1[:, 0], kps_1[:, 1], c='r', s=20)
plt.xlabel('x (pixesl)')
plt.ylabel('y (pixels)')
plt.title('Image 1')
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(img0, cmap='gray')
plt.scatter(kps_2[:, 0], kps_2[:, 1], c='r', s=20)
plt.xlabel('x (pixesl)')
plt.ylabel('y (pixels)')
plt.title('Image 2')
plt.show()

# 3D plot of the 3D landmarks (X)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', s=20)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('3D landmarks (X)')
plt.show()

