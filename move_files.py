import os
import shutil

# Define the source directory where your folders and subfolders are
source_directory = "E:\Stuff\Python\Video Retrieval\clip_model\Projects\Competition\DB"

# Define the destination directory where you want to copy the images
destination_directory = "E:\Stuff\Python\Video Retrieval\clip_model\Projects\Competition\Database"

# Create the destination directory if it doesn't already exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Walk through the source directory
for dirpath, dirnames, filenames in os.walk(source_directory):
    for file in filenames:
        # Check if the file is an image (you can add more extensions if needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Create the full path to the source file
            file_path = os.path.join(dirpath, file)
            # Copy the file to the destination directory
            shutil.copy(file_path, destination_directory)

print('Images have been copied to:', destination_directory)
