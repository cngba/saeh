import os
import pickle
import numpy as np
from PIL import Image

# Path to CIFAR-10 dataset batches
cifar10_batches_path = os.path.expandvars(r"%USERPROFILE%\.keras\datasets\cifar-10-batches-py-target\cifar-10-batches-py")
output_folder = os.path.join(os.path.dirname(__file__), 'all_images')

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to unpickle CIFAR-10 batch files
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# Extract images from data_batch_1 to data_batch_5
for batch_id in range(1, 6):
    batch_file = os.path.join(cifar10_batches_path, f"data_batch_{batch_id}")
    if not os.path.exists(batch_file):
        print(f"Batch file '{batch_file}' not found. Skipping...")
        continue

    print(f"Processing {batch_file}...")
    batch_data = unpickle(batch_file)
    images = batch_data[b'data']
    labels = batch_data[b'labels']

    # Reshape and save images
    for i, img_data in enumerate(images):
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)  # Convert to (32, 32, 3)
        img = Image.fromarray(img)
        img_index = (batch_id - 1) * 10000 + i  # Calculate the image index
        img_filename = os.path.join(output_folder, f"{img_index}.png")
        img.save(img_filename)

print(f"Images extracted and saved to '{output_folder}'.")