import cv2
import os

# Set the path to the directory containing your images
image_folder = 'images_malaga'

# Set the desired video name and codec
video_name = 'malaga_dataset_video.mp4'
codec = cv2.VideoWriter_fourcc(*'mp4v')

# Set the frames per second (fps) for the video
fps = 20

# Get the list of image files in the specified directory
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# Sort the images based on the numeric part of the filenames
images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
print(f"images: {images}")

# Determine the dimensions of the first image
img = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = img.shape

# Create a VideoWriter object
video = cv2.VideoWriter(video_name, codec, fps, (width, height))

# Iterate through the images and write each frame to the video
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    video.write(img)

# Release the VideoWriter object
video.release()

print("Video created successfully!")