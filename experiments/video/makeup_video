import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import cv2
import glob

# Set frame rate and output video file name
tag = "tag"
folder_path = "testing/"+tag
output_video = "output.mp4"
folder_path =  folder_path+'_'+ "combined"  # Folder to save merged images

# Get all PNG images in the current directory (sorted by filename)
image_files = sorted(glob.glob(folder_path+'//'+"*.png"))


fps = 1  

# Read the first image to determine video resolution
first_frame = cv2.imread(image_files[0])
height, width, layers = first_frame.shape

# Define video codec (MP4 format using XVID codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write each image frame to the video
for img in image_files:
    frame = cv2.imread(img)
    video.write(frame)

# Release the video writer
video.release()
print(f"✅ Video saved as {output_video}")

