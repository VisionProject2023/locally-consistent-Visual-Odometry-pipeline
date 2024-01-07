import os
from os.path import expanduser
from pathlib import Path

from copy import deepcopy
from random import randint
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Visual():

    def _init_(self, K):
        self._im = None
        self._landmarks = []
        self._landmarks_px = []
        self._position_history = []
        self._number_of_keypoints = []
        self._tracked_px = []
        self._H_latest = np.eye(4)
        self._K = K
        self._iter = 2
        self._fig = plt.figure(figsize=(36, 18))

    def within_image(self, uv, img_dims):
        """Given xy pixel coordinate and the img shape (width, height),
        return True if pixel is inside the image bounds"""
        return 0 <= uv[0] <= img_dims[0] and 0 <= uv[1] <= img_dims[1]

    def update(self, im, state, H):
        """
        Update visualization based on current state and H matrix found 
        """
        # Update image
        self._im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Store keypoints
        self._landmarks_px = []
        self._landmarks_px.append(state['P'])
        self._landmarks = []
        self._landmarks.append(state['X'])
        self._number_of_keypoints.append(state['X'].shape[0])

        # Store trajectory
        H = H
        axis = axis = np.linalg.inv(H) @ np.vstack((np.hstack((np.eye(3), np.zeros((3,1)))), np.ones((4,1)).T))
        self._position_history.append(axis[:,3])

        # Store pose for projecting landmarks
        self._H_latest = H
        self._iter += 1


    def render(self):
        self._fig = plt.figure(figsize=(12, 6))
        landmarks= np.array(self._landmarks[0])
        filter = np.linalg.norm(landmarks, axis = 1) < 200
        ax = self._fig.add_subplot(221)
        plt.title("Landmarks and Keypoints img {}".format(self._iter))
        ax.imshow(self._im)

        # Draw keypoints
        landmarks_px = np.array(self._landmarks_px[0])
        ax.scatter(landmarks_px[filter, 0], landmarks_px[filter, 1], s=4, c='green', facecolor=None)
        ax.set_xlim([0, self._im.shape[1]])
        ax.set_ylim([self._im.shape[0], 0])
        ax.legend(["Landmarks"], loc='lower right')
        plt.xticks([])
        plt.yticks([])

        # # Plot Landmark history
        # ax = self._fig.add_subplot(245)
        # plt.title("# Landmarks")
        # ax.plot(list(range(len(self._landmark_history))), self._landmark_history)


        ax = self._fig.add_subplot(122)
        traj_len = len(self._position_history)
        traj = np.vstack(self._position_history)
        plt.title("Local Trajectory")
        ax.scatter(traj[max([0, traj_len-20]):, 0], traj[max([0, traj_len-20]):, 2], s=20, c='blue', facecolor=None)
        ax.set_xlim(np.min(traj[max([0, traj_len-20]):,0]) -2, np.max(traj[max([0, traj_len-20]):,0]) + 2)
        ax.set_ylim(np.min(traj[max([0, traj_len-20]):,2]) -2, np.max(traj[max([0, traj_len-20]):,2]) + 2)
        ax.set_aspect("equal")
        ax.set_adjustable("datalim")
        # Draw landmarks in map
        ax.scatter(landmarks[filter, 0], landmarks[filter, 2], s=20, c='green', facecolor=None, alpha=0.2)

        ax = self._fig.add_subplot(246)
        plt.title("Number_of_keypoints")
        len_plot = min(len(self._number_of_keypoints), 20)
        starting = len(self._number_of_keypoints) - len_plot
        indices = np.arange(starting, len(self._number_of_keypoints), 1)
        print("indices" ,indices)
        ax.plot(indices, self._number_of_keypoints[starting:])
        ax.set_adjustable("datalim")

        # Draw trajectory
        ax = self._fig.add_subplot(245)
        plt.title("Global Trajectory")
        for axis in self._position_history:
            # print("axis ", axis)
            plt.scatter(axis[0], axis[2], s=20, c='blue', facecolor=None)
        ax.set_aspect("equal")
        ax.set_adjustable("datalim")
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)


        self._fig.canvas.draw()
        im_vis = np.fromstring(self._fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        im_vis = im_vis.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)

        
        #cv2.imshow("Visualization", im_vis)
        file_name = "images/image_" + str(self._iter)
        plt.savefig(file_name)
        plt.close(self._fig)