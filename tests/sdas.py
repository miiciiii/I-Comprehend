import numpy as np
import os

# Base directory and label path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LABEL_DIR = os.path.join(BASE_DIR, 'datasets', 'processed', '2labels', 'labels_chunk_0.npy')

# Load the label encoders
arousal_classes = np.load(os.path.join(BASE_DIR, 'datasets', 'processed', 'arousal_encoder.npy'))
dominance_classes = np.load(os.path.join(BASE_DIR, 'datasets', 'processed', 'dominance_encoder.npy'))

# Mappings for the string representation of classes
arousal_mapping = {
    0: 'Calm',
    1: 'Dull',
    2: 'Excited',
    3: 'Neutral',
    4: 'Wide-awake'
}

dominance_mapping = {
    0: 'Dependent',
    1: 'Independent',
    2: 'Neutral',
    3: 'Powerful',
    4: 'Powerlessness'
}

# Function to convert labels to numeric
def convert_labels_to_numeric(labels):
    try:
        # Convert first two labels to integers
        arousal_label = int(labels[0])
        dominance_label = int(labels[1])
        
        # Validate the indices
        if arousal_label < 0 or arousal_label >= len(arousal_mapping):
            raise ValueError(f"Arousal index {arousal_label} out of bounds.")
        if dominance_label < 0 or dominance_label >= len(dominance_mapping):
            raise ValueError(f"Dominance index {dominance_label} out of bounds.")
        
        # Ensure remaining features are also numeric
        return [arousal_label, dominance_label] + list(map(int, labels[2:]))
    except (KeyError, ValueError) as e:
        print(f"Error with label {labels}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error converting labels {labels}: {e}")
        return None

# Load the labels from the .npy file
y = np.load(LABEL_DIR, mmap_mode='r').astype('float32')

# Display some samples before conversion
num_samples_to_display = 5  # Number of samples to display
for i in range(min(num_samples_to_display, len(y))):
    print(f"Sample {i} before conversion: {y[i]}")

if np.issubdtype(y.dtype, np.number):
        print("Labels are already in numeric format.")
        # Ensure that y only contains numeric labels for the model
        y = y.astype(int)  # Ensure it's of integer type

# Convert labels to numeric
y = np.array([convert_labels_to_numeric(label) for label in y if convert_labels_to_numeric(label) is not None])

# Print the converted labels
print("\nConverted labels (y):")
print(y)


# # EDA - Check shape of labels
# print(f"Shape of label data: {y.shape}")

# # Extract each class for analysis
# arousal_labels = y[:, 0]
# dominance_labels = y[:, 1]
# effort_labels = y[:, 2]
# frustration_labels = y[:, 3]
# mental_demand_labels = y[:, 4]
# performance_labels = y[:, 5]
# physical_demand_labels = y[:, 6]

# # Helper function to plot label distribution
# def plot_distribution(label_array, class_mapping, class_name):
#     label_counts = Counter(label_array)
#     labels, counts = zip(*sorted(label_counts.items()))

#     # If we have a mapping, convert the indices to string labels
#     if class_mapping:
#         labels = [class_mapping[label] for label in labels]
    
#     plt.figure(figsize=(8, 5))
#     sns.barplot(x=labels, y=counts, palette='viridis')
#     plt.title(f'{class_name} Class Distribution')
#     plt.ylabel('Frequency')
#     plt.xlabel('Class')
#     plt.xticks(rotation=45)
#     plt.show()

# # Plot distribution of each class
# plot_distribution(arousal_labels, arousal_mapping, 'Arousal')
# plot_distribution(dominance_labels, dominance_mapping, 'Dominance')
# plot_distribution(effort_labels, None, 'Effort')
# plot_distribution(frustration_labels, None, 'Frustration')
# plot_distribution(mental_demand_labels, None, 'Mental Demand')
# plot_distribution(performance_labels, None, 'Performance')
# plot_distribution(physical_demand_labels, None, 'Physical Demand')

# # Data Integrity Check - Ensure all labels are within the correct range
# def check_label_ranges(labels, max_value, class_name):
#     invalid_labels = labels[(labels < 0) | (labels >= max_value)]
#     if len(invalid_labels) > 0:
#         print(f"Warning: {len(invalid_labels)} invalid {class_name} labels found. Invalid labels: {invalid_labels}")
#     else:
#         print(f"All {class_name} labels are valid.")

# # Checking the validity of all labels
# check_label_ranges(arousal_labels, len(arousal_mapping), 'Arousal')
# check_label_ranges(dominance_labels, len(dominance_mapping), 'Dominance')
# check_label_ranges(effort_labels, num_classes, 'Effort')
# check_label_ranges(frustration_labels, num_classes, 'Frustration')
# check_label_ranges(mental_demand_labels, num_classes, 'Mental Demand')
# check_label_ranges(performance_labels, num_classes, 'Performance')
# check_label_ranges(physical_demand_labels, num_classes, 'Physical Demand')
