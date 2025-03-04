import cv2
import os
import glob

tag = "tag"
folder_path = "testing/"+tag
agent_path = "_agent_1"
top_view_path = "_top_view"
# Define input directories
dir_A = folder_path+agent_path  # Left side images
dir_B = folder_path+top_view_path  # Right side images
output_dir =  folder_path+'_'+ "combined"  # Folder to save merged images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of image filenames from both directories (sorted to ensure matching)
images_A = sorted(glob.glob(os.path.join(dir_A, "*.png")))
images_B = sorted(glob.glob(os.path.join(dir_B, "*.png")))

# Ensure both directories have the same number of images
if len(images_A) != len(images_B):
    raise ValueError("The number of images in both directories does not match!")

# Iterate through images and merge them
for img_A, img_B in zip(images_A, images_B):
    # Read images
    image_A = cv2.imread(img_A)
    image_B = cv2.imread(img_B)

    # Ensure images have the same height
    if image_A.shape[0] != image_B.shape[0]:
        height = min(image_A.shape[0], image_B.shape[0])
        image_A = cv2.resize(image_A, (image_A.shape[1], height))
        image_B = cv2.resize(image_B, (image_B.shape[1], height))

    # Concatenate images side by side
    combined_image = cv2.hconcat([image_A, image_B])

    # Save merged image
    output_filename = os.path.join(output_dir, os.path.basename(img_A))
    cv2.imwrite(output_filename, combined_image)

print(f"✅ Merged images saved in '{output_dir}' folder.")
