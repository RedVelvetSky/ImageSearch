import os

import numpy as np
import open_clip
import torch
from PIL import Image

# Check if CUDA is available and load the model onto the GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the pre-trained CLIP model and preprocessing function
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device).eval()  # Move the model to the GPU and set it to evaluation mode


def encode_images_in_batches(image_folder, batch_size=32, output_file='features_sorted.npy'):
    # List images in the folder
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort files to maintain consistent order

    all_features = []  # This will store all features as a list first

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []

        for image_path in batch_files:
            image = Image.open(image_path).convert('RGB')
            image = preprocess(image)  # Preprocess the image
            batch_images.append(image)

        batch_images = torch.stack(batch_images).to(device)  # Stack and move to GPU
        with torch.no_grad():
            batch_features = model.encode_image(batch_images)  # Forward pass for the batch
            all_features.extend(batch_features.cpu().numpy())  # Move features to CPU and convert to numpy array

    # Convert list of numpy arrays into a single numpy array
    all_features = np.vstack(all_features)
    # Save features as a compressed file
    np.save(output_file, all_features)

    print(f"Completed encoding all images. Data saved to {output_file}.")
    return output_file


def encode_images_in_batches_from_list(image_files, batch_size=32, output_file='features_selected.npy'):
    """
    Encode images provided in a list using a specified model and save the features.

    Args:
    image_files (list): List of image file paths.
    batch_size (int): Number of images to process in each batch.
    output_file (str): Path to save the numpy array of features.
    preprocess (callable): Function to preprocess images.
    model (torch model): Model to encode images.
    device (torch device): Device on which to perform computation (e.g., 'cuda:0').

    Returns:
    str: Path to the output file where features are saved.
    """
    all_features = []  # This will store all features as a list first

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []

        for image_path in batch_files:
            image = Image.open(image_path).convert('RGB')
            image = preprocess(image)  # Preprocess the image
            batch_images.append(image)

        if not batch_images:  # In case of an empty list
            continue

        # print("batch images:", batch_images)
        batch_images = torch.stack(batch_images).to(device)  # Stack and move to GPU
        with torch.no_grad():
            batch_features = model.encode_image(batch_images)  # Forward pass for the batch
            all_features.extend(batch_features.cpu().numpy())  # Move features to CPU and convert to numpy array

    # Convert list of numpy arrays into a single numpy array
    all_features = np.vstack(all_features)
    # Save features as a compressed file
    np.save(output_file, all_features)

    print(f"Completed encoding all images. Data saved to {output_file}.")
    return output_file


# Example usage
image_folder = "D:\\Stuff\\Database\\"
output_file = "D:\\Stuff\\features.npy"


# encode_images_in_batches(image_folder, batch_size=512, output_file=output_file)

def load_features(file_path):
    # Load the numpy array
    features_array = np.load(file_path)
    # Convert the numpy array to a PyTorch tensor
    features_tensor = torch.from_numpy(
        features_array).float()  # Ensure the tensor is of type float for any further computation
    return features_tensor

# Example usage
# features_tensor = load_features("D:\\Stuff\\features.npy")
# print(features_tensor.shape)
# print(features_tensor[0])
