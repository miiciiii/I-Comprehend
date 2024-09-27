import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Paths to the saved chunks
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_CHUNK_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', '2images', 'images_chunk_0.npy')  # Load first chunk
LABEL_CHUNK_PATH = os.path.join(BASE_DIR, 'datasets', 'processed', '2labels', 'labels_chunk_0.npy')  # Load first chunk

# Load image and label chunks
print("Loading image and label data...")
image_data = np.load(IMAGE_CHUNK_PATH, mmap_mode= 'r')
labels_data = np.load(LABEL_CHUNK_PATH, mmap_mode= 'r')
print(f"Loaded {image_data.shape[0]} images and {labels_data.shape[0]} labels.")

# 1. Image Data Analysis

# a. Display some random images
def display_images(images, num_images=5):
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.suptitle(f"Displaying {num_images} sample images")
    plt.show()

display_images(image_data)

# b. Image dimensions and pixel value range
print(f"Image shape: {image_data.shape}")  # (num_images, height, width, channels)
print(f"Image data type: {image_data.dtype}")
print(f"Pixel value range: {image_data.min()} to {image_data.max()}")  # Since images are normalized to [0, 1]

# c. Pixel value distribution
plt.figure(figsize=(8, 6))
plt.hist(image_data.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Pixel Value Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# d. Image statistics (mean, std)
mean_pixel_value = image_data.mean()
std_pixel_value = image_data.std()
print(f"Mean pixel value: {mean_pixel_value:.4f}")
print(f"Standard deviation of pixel values: {std_pixel_value:.4f}")

# 2. Label Data Analysis

# Assuming label columns: [arousal, dominance, effort, frustration, mental_demand, performance, physical_demand]
label_names = ['arousal', 'dominance', 'effort', 'frustration', 'mental_demand', 'performance', 'physical_demand']

# a. Label distributions (categorical labels: arousal and dominance)
arousal_distribution = Counter(labels_data[:, 0])
dominance_distribution = Counter(labels_data[:, 1])

def plot_label_distribution(label_name, distribution):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(distribution.keys()), y=list(distribution.values()))
    plt.title(f'Distribution of {label_name}')
    plt.xlabel(f'{label_name.capitalize()} Class')
    plt.ylabel('Count')
    plt.show()

plot_label_distribution('Arousal', arousal_distribution)
plot_label_distribution('Dominance', dominance_distribution)

# b. Other continuous labels (effort, frustration, etc.)
# Plot histograms for these labels
for i, label_name in enumerate(label_names[2:]):  # Skip arousal and dominance
    plt.figure(figsize=(6, 4))
    plt.hist(labels_data[:, i + 2], bins=20, color='green', alpha=0.7)
    plt.title(f'Distribution of {label_name.capitalize()}')
    plt.xlabel(label_name.capitalize())
    plt.ylabel('Frequency')
    plt.show()

# c. Correlation between labels (if relevant)
# Compute and display a correlation matrix between the continuous labels
continuous_labels = labels_data[:, 2:].astype(float)  # Only effort, frustration, mental_demand, etc.
corr_matrix = np.corrcoef(continuous_labels, rowvar=False)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=label_names[2:], yticklabels=label_names[2:])
plt.title('Correlation Matrix of Continuous Labels')
plt.show()
