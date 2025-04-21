import keras
from keras.datasets import cifar10, mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, CSVLogger
from keras.models import Model, load_model
from keras import optimizers
import matplotlib.pyplot as plt
from PIL import Image
import os
import psutil
import time
import numpy as np
import pandas as pd
import hash_model
import load_data
import argparse
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


TF_ENABLE_ONEDNN_OPTS=0

which_data = "cifar10"  
num_classes = 10       
stack_num = 18    # number of stack in the ResNet network
batch_size = 64   # number of training batch per step
epochs = 10       # number of training epoch

# weight in the loss function
alpha = 1e-1    # weight of binary loss term
beta = 1e-1     # weight of evenly distributed term
gamma = 1   # weight of recovery loss term

hash_bits = 64  # length of hash bits to encode

base = './saved/'+which_data+'/SAEH/'  # model and log path to be saved
load_path = ""
save_path = base + "data/"
log_path = base + "log/"

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

# changing weight scheduler, we change the learning weight according to the epoch
def scheduler(epoch):
    if epoch <= 30:
        return 0.1
    if epoch <= 55:
        return 0.02
    if epoch <= 75:
        return 0.004
    return 0.0008

# Function to save hash table to a CSV file
def save_hash_table(model, data, file_path="hash_table_" + str(hash_bits) + ".csv"):
    # Extract hash codes
    hash_layer_model = Model(inputs=model.input, outputs=model.get_layer('hash_x').output)
    hash_codes = hash_layer_model.predict(data)
    binary_hash_codes = (hash_codes > 0.5).astype(int)  # Convert to binary

    # Save to CSV
    df = pd.DataFrame(binary_hash_codes)
    df.to_csv(file_path, index=False, header=False)
    print(f"Hash table saved to '{file_path}'")

# Function to plot the most similar images
def plot_similar_images(image_folder, hash_table, index, nearest_indices):
    input_image_path = os.path.join(image_folder, f'{index}.png')  # Assuming images are named as index.png
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image '{input_image_path}' not found in folder '{image_folder}'.")

    # Load and plot the input image
    input_image = Image.open(input_image_path)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, len(nearest_indices) + 1, 1)
    plt.imshow(input_image)
    plt.title(f"Input Image\n{index}")
    plt.axis('off')

    # Plot nearest images
    for i, nearest_index in enumerate(nearest_indices):
        nearest_image_path = os.path.join(image_folder, f'{nearest_index}.png')
        if not os.path.exists(nearest_image_path):
            raise FileNotFoundError(f"Nearest image '{nearest_image_path}' not found in folder '{image_folder}'.")

        nearest_image = Image.open(nearest_image_path)
        plt.subplot(1, len(nearest_indices) + 1, i + 2)
        plt.imshow(nearest_image)
        plt.title(f"Nearest {i + 1}\n{nearest_index}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'nearest_images_{index}.png'))
    print(f"Nearest images plotted and saved as 'nearest_images_{index}.png' in {save_path}")
    plt.show()
    

# Function to retrieve and plot nearest images
def retrieve_and_plot_nearest(image_folder, hash_table, index, top_k=11):
    nearest_indices, nearest_distances = retrieve_nearest(hash_table, index, top_k=top_k)
    print(f"Nearest {top_k} hashes for index {index}:")
    for i, (nearest_index, dist) in enumerate(zip(nearest_indices, nearest_distances)):
        print(f"{i + 1}. Index: {nearest_index}, Hamming Distance: {dist:.4f}")

    plot_similar_images(image_folder, hash_table, index, nearest_indices)

# Function to load the hash table from a CSV file
def load_hash_table(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hash table file '{file_path}' not found.")
    return pd.read_csv(file_path, header=None).values

# Function to retrieve the nearest hashes and their Hamming distances
def retrieve_nearest(hash_table, index, top_k=11):
    if index < 0 or index >= len(hash_table):
        raise ValueError(f"Index {index} is out of bounds for the hash table.")
    
    query_hash = hash_table[index].reshape(1, -1)  # Get the hash for the given index
    distances = cdist(query_hash, hash_table, metric='hamming')[0]  # Compute Hamming distances
    nearest_indices = distances.argsort()[:top_k]  # Get indices of the nearest hashes
    nearest_distances = distances[nearest_indices]  # Get the corresponding distances
    return nearest_indices, nearest_distances

# Function to log memory usage
def log_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"[{stage}] Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")

# Function to measure execution time
def time_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Execution Time for {func.__name__}: {end_time - start_time:.2f} seconds")
    return result

# Function to evaluate the model
def evaluate_model(model, x_test, y_test, hash_table_path, top_k=10):
    print("\nStarting Evaluation...")
    log_memory_usage(stage="Evaluation Start")

    # Predict outputs
    y_predict, y_decoded = model.predict(x_test, batch_size=64)

    # Reconstruction Loss
    reconstruction_loss = np.mean(np.square(x_test - y_decoded))
    print(f"Reconstruction Loss (MSE): {reconstruction_loss}")

    # Classification Accuracy
    y_pred_classes = np.argmax(y_predict, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"Classification Accuracy: {accuracy}")

    # Load Hash Table
    hash_table = load_hash_table(hash_table_path)

    # Evaluate Retrieval Metrics
    subset_size = 100  # Use a subset for faster evaluation
    hash_table_subset = hash_table[:subset_size]
    y_true_classes_subset = y_true_classes[:subset_size]

    precision_at_k_values = []
    average_precisions = []
    for i in range(len(hash_table_subset)):
        query_hash = hash_table_subset[i]
        distances = np.sum(query_hash != hash_table, axis=1) / hash_table.shape[1]
        sorted_indices = np.argsort(distances)[:top_k]

        relevant_indices = np.where(y_true_classes == y_true_classes_subset[i])[0]
        retrieved_indices = sorted_indices

        precision_at_k = len(set(relevant_indices) & set(retrieved_indices)) / top_k
        precision_at_k_values.append(precision_at_k)

        precisions = []
        for k in range(1, top_k + 1):
            if retrieved_indices[k - 1] in relevant_indices:
                precisions.append(len(set(relevant_indices) & set(retrieved_indices[:k])) / k)
        if precisions:
            average_precisions.append(np.mean(precisions))

    mean_precision_at_k = np.mean(precision_at_k_values)
    mean_average_precision = np.mean(average_precisions)
    print(f"Mean Precision@{top_k}: {mean_precision_at_k}")
    print(f"Mean Average Precision (mAP): {mean_average_precision}")

    log_memory_usage(stage="Evaluation End")
    return {
        "reconstruction_loss": reconstruction_loss,
        "classification_accuracy": accuracy,
        "mean_precision_at_k": mean_precision_at_k,
        "mean_average_precision": mean_average_precision
    }

def plot_training_metrics(training_log_path):
    # Load the training log
    df = pd.read_csv(training_log_path)

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['loss'], label='Training Loss', color='blue')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    if 'y_predict_accuracy' in df.columns and 'val_y_predict_accuracy' in df.columns:
        plt.plot(df['epoch'], df['y_predict_accuracy'], label='Training Accuracy', color='blue')
        plt.plot(df['epoch'], df['val_y_predict_accuracy'], label='Validation Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
    else:
        print("Accuracy columns not found in training log. Skipping accuracy plot.")

    # Save and Show the Plot
    plot_path = os.path.join(os.path.dirname(training_log_path), "training_metrics.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training metrics plot saved to '{plot_path}'")
    plt.show()

# Refine the ranking of the retrieved images using feature similarity (cosine).
def refine_ranking(model, image_data, hash_table, index, nearest_indices, top_k=11, feature_layer='hash_x'):
    """
    Parameters:
    - model: The trained Keras model containing the encoder.
    - image_data: The complete dataset of input images (e.g., x_train).
    - hash_table: The hash table for fast lookup (not directly used here, but kept for compatibility).
    - index: Index of the query image in the dataset.
    - nearest_indices: Initial nearest indices obtained from hash-based retrieval.
    - top_k: Number of top images to return after re-ranking.
    - feature_layer: Layer name from which to extract features for re-ranking.

    Returns:
    - refined_indices: Re-ranked indices based on feature similarity.
    """

    # Create a model to extract intermediate features from the encoder
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(feature_layer).output)

    # Extract features for the query image
    query_image = image_data[index:index+1]  # Shape (1, H, W, C)
    query_feature = feature_extractor.predict(query_image)  # Shape (1, D)

    # Extract features for all initially retrieved images
    retrieved_images = image_data[nearest_indices]  # Shape (top_k, H, W, C)
    retrieved_features = feature_extractor.predict(retrieved_images)  # Shape (top_k, D)

    # Compute cosine similarity between query and retrieved features
    similarity_scores = cosine_similarity(query_feature, retrieved_features)[0]  # Shape (top_k,)

    # Sort the retrieved indices based on similarity score in descending order
    sorted_indices = np.argsort(-similarity_scores)  # Highest similarity first
    refined_indices = [nearest_indices[i] for i in sorted_indices[:top_k]]

    return refined_indices


# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Supervised Autoencoder Hashing")
    parser.add_argument('--mode', type=str, choices=['train', 'retrieval', 'evaluation'], default='train',
                        help="Mode of operation: 'train' to train the model, 'retrieval' for hash retrieval, 'evaluation' to evaluate the model.")
    parser.add_argument('--index', type=int, default=0,
                        help="Index of the hash to retrieve nearest neighbors (used in 'retrieval' mode).")
    parser.add_argument('--top_k', type=int, default=11,
                        help="Number of nearest neighbors to retrieve (used in 'retrieval' mode).")
    parser.add_argument('--image_folder', type=str, default='all_images',
                        help="Folder containing the images for retrieval.")
    args = parser.parse_args()

    if args.mode == 'train':
        # ...existing training code...
        (x_train, y_train), (x_test, y_test) = load_data.load_data(which_data)
        (_, img_rows, img_cols, img_channels) = x_train.shape

        hash_su_ae_model = hash_model.HashSupervisedAutoEncoderModel(
            img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits, alpha, beta, gamma
        )
        
        resnet = Model(inputs=hash_su_ae_model.img_input, outputs=[hash_su_ae_model.y_predict, hash_su_ae_model.y_decoded])
        if load_path:
            resnet.load_weights(load_path)

        print(resnet.summary())

        sgd = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        resnet.compile(
            optimizer=sgd,
            loss={'y_predict': 'categorical_crossentropy', 'y_decoded': hash_su_ae_model.net_loss},
            metrics={'y_predict': 'accuracy'},
            loss_weights=[1, 1.],  # Ensure the weight for 'y_decoded' is not 0
        )

        tb_cb = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=False)
        change_lr = LearningRateScheduler(scheduler)
        chk_pt = ModelCheckpoint(filepath=save_path+'chk_pt.h5', monitor='val_y_predict_accuracy', save_best_only=True, mode='max', save_freq='epoch')
        csv_log_path = save_path + "training_log.csv"
        csv_logger = CSVLogger(csv_log_path, append=True)
        cbks = [change_lr, tb_cb, chk_pt, csv_logger]

        resnet.fit(
            x_train, {"y_predict": y_train, "y_decoded": x_train},
            epochs=epochs, batch_size=batch_size, callbacks=cbks,
            validation_data=(x_test, [y_test, x_test])
        )

        resnet.save(save_path+"hash_su_ae.h5")
        print(f"Model saved at '{save_path}hash_su_ae.h5'")

        hash_table_path = save_path + "hash_table_" + str(hash_bits) + ".csv"
        save_hash_table(resnet, x_train, file_path=hash_table_path)

    elif args.mode == 'retrieval':
        # Load test data
        (x_train, y_train), (x_test, y_test) = load_data.load_data(which_data)
        (_, img_rows, img_cols, img_channels) = x_train.shape

        # Load the trained model
        model_path = save_path + "hash_su_ae.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Train the model first.")
        resnet = load_model(model_path, custom_objects={'net_loss': hash_model.HashSupervisedAutoEncoderModel.net_loss})

        hash_table_path = save_path + "hash_table_" + str(hash_bits) + ".csv"
        hash_table = load_hash_table(hash_table_path)

        # First retrieve based on hashing
        nearest_indices, nearest_distances = retrieve_nearest(hash_table, args.index, top_k=args.top_k)

        # Then refine using intermediate features (e.g., cosine similarity)
        refined_indices = refine_ranking(resnet, x_train, hash_table, args.index, nearest_indices, top_k=args.top_k)
        refined_distances = cdist(hash_table[args.index].reshape(1, -1), hash_table[refined_indices], metric='hamming')[0]

        print(f"Refined nearest {args.top_k} hashes for index {args.index}:")
        for i, (idx, dist) in enumerate(zip(refined_indices, refined_distances)):
            print(f"{i + 1}. Index: {idx}, Hamming Distance: {dist:.4f}")

        # Plot using the refined results
        plot_similar_images(args.image_folder, hash_table, args.index, refined_indices)

    elif args.mode == 'evaluation':
        # Load test data
        (x_train, y_train), (x_test, y_test) = load_data.load_data(which_data)
        (_, img_rows, img_cols, img_channels) = x_train.shape

        # Load the trained model
        model_path = save_path + "hash_su_ae.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Train the model first.")
        resnet = load_model(model_path, custom_objects={'net_loss': hash_model.HashSupervisedAutoEncoderModel.net_loss})

        # Evaluate the model
        hash_table_path = save_path + "hash_table_" + str(hash_bits) + ".csv"
        evaluation_results = evaluate_model(resnet, x_test, y_test, hash_table_path)

        print("\nEvaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value}")
        # Plot training metrics
        training_log_path = save_path + "training_log.csv"
        plot_training_metrics(training_log_path)
"""
### Usage:
1. **Training Mode**:
   python hash_su_ae_train.py --mode train

2. **Retrieval Mode**:
   python hash_su_ae_train.py --mode retrieval --index 5 --top_k 11

3. **Evaluation Mode**:
    python hash_su_ae_train.py --mode evaluation --top_k 11
"""