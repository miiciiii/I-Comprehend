import shutil
import os
import requests
import tarfile
from time import sleep

# Define the directory where you want to save the model relative to the script location
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_dir = os.path.join(base_dir, 's2v_old')

# Define the paths for the downloaded and extracted files
tar_file = os.path.join(base_dir, 's2v_reddit_2015_md.tar.gz')
extracted_dir = 's2v_reddit_2015_md'

# Download the model if not already downloaded
url = 'https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz'

def download_file(url, filename, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check if the request was successful
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded {filename} successfully.")
            return
        except requests.RequestException as e:
            print(f"Download failed on attempt {attempt + 1}/{retries}: {e}")
            sleep(5)  # Wait before retrying

download_file(url, tar_file)

# Extract the tar.gz file
with tarfile.open(tar_file, 'r:gz') as tar:
    tar.extractall(path=base_dir)

# Ensure the target directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


print(f"Model has been downloaded")
