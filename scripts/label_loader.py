import os
import pandas as pd
from tqdm import tqdm

def get_txt_files(directory):
    """Retrieve a list of .txt files in the given directory."""
    print(f"Checking directory: {directory}")
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist or is not a directory.")
    
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} .txt files: {txt_files}")
    print("List of .txt files:", txt_files)  # Debugging statement
    return txt_files

def load_labels_from_file(file_path):
    """Load labels from a single .txt file into a dictionary."""
    print(f"Loading file: {file_path}")
    metric = os.path.basename(file_path).split('.')[0]
    print(f"Metric extracted from file name: {metric}")

    # Read the file into a DataFrame
    try:
        df = pd.read_csv(file_path, header=None, names=['task', 'label'])
        print(f"Loaded DataFrame with {len(df)} rows from {file_path}")
        print("DataFrame head:", df.head())  # Debugging statement
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

    # Ensure that the expected columns exist
    if 'task' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"The file {file_path} does not contain the required columns 'task' and 'label'.")
    
    labels = {}
    for _, row in df.iterrows():
        task, label = row['task'], row['label']
        print(f"Processing row - Task: {task}, Label: {label}")
        if task not in labels:
            labels[task] = {}
        labels[task][metric] = label
    
    print(f"Labels loaded from {file_path}: {labels}")
    return labels

def load_labels(ground_truths_path="datasets/raw/ground_truths"):
    """Load labels from all .txt files in the specified directory."""
    print(f"Loading labels from directory: {ground_truths_path}")
    if not os.path.isdir(ground_truths_path):
        raise ValueError(f"The directory {ground_truths_path} does not exist or is not a directory.")
    
    all_labels = {}
    txt_files = get_txt_files(ground_truths_path)
    
    for file in tqdm(txt_files, desc="Loading Labels"):
        file_path = os.path.join(ground_truths_path, file)
        print(f"Processing file: {file}")
        file_labels = load_labels_from_file(file_path)
        
        for task, metrics in file_labels.items():
            print(f"Updating labels for task: {task} with metrics: {metrics}")
            if task not in all_labels:
                all_labels[task] = {}
            all_labels[task].update(metrics)
            print(f"Current state of labels for task {task}: {all_labels[task]}")
    
    print(f"All labels loaded: {all_labels}")
    print("Final labels:", all_labels)  # Debugging statement
    return all_labels