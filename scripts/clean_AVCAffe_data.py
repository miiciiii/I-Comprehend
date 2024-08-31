import os
import shutil

# Paths for extraction and final destination
base_path = r"D:\03PersonalFiles\Thesis\I-Comprehend\datasets\raw"
extracted_path = os.path.join(base_path, "extracted_file")
final_path = os.path.join(base_path, "final_face_crops")

# Create directories if they don't exist
os.makedirs(extracted_path, exist_ok=True)
os.makedirs(final_path, exist_ok=True)

# Iterate over each directory
for dir_name in os.listdir(os.path.join(base_path, "face_crops")):
    curr_path = os.path.join(base_path, "face_crops", dir_name)

    if os.path.isdir(curr_path):
        for clip in os.listdir(curr_path):
            name = clip[:-4]  # Remove the .zip extension
            folder_name = name[:14]
            file_path = os.path.join(curr_path, clip)
            send_to = os.path.join(extracted_path, folder_name)

            print(f"Processing folder: {folder_name}")

            if file_path.endswith('.zip'):
                # Create a directory for extraction if it doesn't exist
                os.makedirs(send_to, exist_ok=True)

                # Extract the zip file
                shutil.unpack_archive(file_path, send_to, 'zip')

                # If the extracted content is another zip file, extract it
                extracted_zip = os.path.join(send_to, clip)
                if os.path.isfile(extracted_zip) and extracted_zip.endswith('.zip'):
                    final_extract_dir = os.path.join(extracted_path, folder_name)
                    os.makedirs(final_extract_dir, exist_ok=True)
                    shutil.unpack_archive(extracted_zip, final_extract_dir, 'zip')

                    # Remove the intermediate zip file
                    os.remove(extracted_zip)

        # Move the specific folder to the final destination        
        final_folder = os.path.join(final_path, folder_name)
        os.makedirs(final_folder, exist_ok=True)

        original_path = os.path.join(extracted_path, folder_name, "shorter_segments_face", name[:7], name[8:14], name)
        
        if os.path.isdir(original_path):
            # Copy all files from the original path to the final folder
            for item in os.listdir(original_path):
                src_file = os.path.join(original_path, item)
                dst_file = os.path.join(final_folder, item)
                shutil.copy(src_file, dst_file)
        else:
            print(f"Original path does not exist: {original_path}")


print("Processing complete.")
