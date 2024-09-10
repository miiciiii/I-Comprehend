import os
import pandas as pd
from tqdm import tqdm

def get_txt_files(directory):
    """Retrieve a list of .txt files in the given directory."""
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist or is not a directory.")
    return [f for f in os.listdir(directory) if f.endswith('.txt')]

def load_labels_from_file(file_path):
    """Load labels from a single .txt file into a dictionary."""
    metric = os.path.basename(file_path).split('.')[0]
    df = pd.read_csv(file_path, header=None, names=['task', 'label'])
    
    # Ensure that the expected columns exist
    if 'task' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"The file {file_path} does not contain the required columns 'task' and 'label'.")
    
    labels = {}
    for _, row in df.iterrows():
        task, label = row['task'], row['label']
        if task not in labels:
            labels[task] = {}
        labels[task][metric] = label
    return labels

def load_labels(ground_truths_path="datasets/raw/ground_truths"):
    """Load labels from all .txt files in the specified directory."""
    if not os.path.isdir(ground_truths_path):
        raise ValueError(f"The directory {ground_truths_path} does not exist or is not a directory.")
    
    all_labels = {}
    txt_files = get_txt_files(ground_truths_path)
    
    for file in tqdm(txt_files, desc="Loading Labels"):
        file_path = os.path.join(ground_truths_path, file)
        file_labels = load_labels_from_file(file_path)
        for task, metrics in file_labels.items():
            if task not in all_labels:
                all_labels[task] = {}
            all_labels[task].update(metrics)
    
    return all_labels
