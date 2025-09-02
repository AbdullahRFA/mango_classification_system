import os
import cv2

# Input folder containing original dataset
src_folder = "data/mango_dataset"

# Output folder where resized images will be stored
dst_folder = "data/mango_dataset_resized"

# Create output folder if it doesn't exist
os.makedirs(dst_folder, exist_ok=True)

# Target image size (width, height) for resizing
target_size = (224, 224)

# Loop through each mango variety folder in the source dataset
for variety in os.listdir(src_folder):
    variety_path = os.path.join(src_folder, variety)

    # Skip if the current item is not a folder (e.g., a file like .DS_Store)
    if not os.path.isdir(variety_path):
        continue

    # Create a matching folder in the destination for this variety
    dst_variety_path = os.path.join(dst_folder, variety)
    os.makedirs(dst_variety_path, exist_ok=True)

    # Loop through each image inside the current variety folder
    for img_name in os.listdir(variety_path):
        img_path = os.path.join(variety_path, img_name)

        # Skip files that are not images (to avoid errors)
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Read the image
        img = cv2.imread(img_path)

        # If the image couldn't be read, skip it
        if img is None:
            continue

        # Resize the image to the target resolution
        img_resized = cv2.resize(img, target_size)

        # Save the resized image into the corresponding destination folder
        cv2.imwrite(os.path.join(dst_variety_path, img_name), img_resized)

# Print success message when done
print("✅ Dataset resized successfully!")