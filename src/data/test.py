import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define paths to load the output data
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LABELS_OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets', 'processed', 'labels')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'datasets', 'processed', 'outputs')

# Create the outputs directory if it doesn't exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Load label chunks
labels = []

# Check if the directory exists
if not os.path.exists(LABELS_OUTPUT_DIR):
    raise FileNotFoundError(f"Labels output directory does not exist: {LABELS_OUTPUT_DIR}")

# Load label chunks
for file in os.listdir(LABELS_OUTPUT_DIR):
    if file.endswith('.npy'):
        label_data = np.load(os.path.join(LABELS_OUTPUT_DIR, file), allow_pickle=True)
        labels.append(label_data)

# Concatenate all loaded label data
labels = np.concatenate(labels, axis=0)  # Combine all chunks

# EDA
print(f"Labels Shape: {labels.shape}")
print(f"Data Type of Labels: {labels.dtype}")

# 3. Analyze Labels
# Convert labels to a DataFrame for easier analysis
labels_df = pd.DataFrame(labels, columns=['arousal', 'dominance', 'effort', 'frustration', 'mental_demand', 'performance', 'physical_demand'])

# Display the first few rows of the labels
print(labels_df.head())

# 4. Summary statistics of labels
print(labels_df.describe())

# 5. Count of each label (count unique values for categorical labels)
for column in labels_df.columns:
    print(f"Counts for {column}:")
    print(labels_df[column].value_counts())

# 6. Visualize label distribution
label_counts = labels_df.apply(pd.Series.value_counts).fillna(0)

# Create a bar plot for label distribution and save it
plt.figure(figsize=(10, 5))
label_counts.plot(kind='bar')
plt.title('Label Distribution')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig(os.path.join(OUTPUTS_DIR, 'label_distribution.png'))
plt.close()  # Close the figure

# 7. Check for Class Imbalance
labels_melted = labels_df.melt(var_name='label_type', value_name='value')

# Create a count plot for class imbalance and save it
plt.figure(figsize=(10, 5))
sns.countplot(data=labels_melted, x='value', hue='label_type')
plt.title('Class Imbalance Across Different Labels')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig(os.path.join(OUTPUTS_DIR, 'class_imbalance.png'))
plt.close()  # Close the figure

print("Plots saved to the outputs directory.")
