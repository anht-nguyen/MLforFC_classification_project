import os
from scipy.io import loadmat
from scripts.utils import get_files_by_class
from scripts.plotting import plot_heatmap
from scripts.config import FC_DATA_PATH, PSD_DATA_PATH, FC_DATASET_ZIP_FILENAME, PSD_DATASET_ZIP_FILENAME

import shutil
import zipfile

# Ensure the data folder exists
data_folder = os.path.join(os.getcwd(), 'data')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print(f"📁 Created data folder: {data_folder}")

    # Move zip files from dataset_zip to data and unzip them
    dataset_zip_folder = os.path.join(os.getcwd(), 'dataset_zip')
    if os.path.exists(dataset_zip_folder):
        for file_name in os.listdir(dataset_zip_folder):
            # print(f"📦 Processing file: {file_name}")
            if file_name[:-4] == FC_DATASET_ZIP_FILENAME or file_name[:-4] == PSD_DATASET_ZIP_FILENAME:
                if file_name.endswith('.zip'):
                    zip_file_path = os.path.join(dataset_zip_folder, file_name)
                    shutil.copy(zip_file_path, data_folder)
                    print(f"📦 Copied zip file: {zip_file_path} to {data_folder}")

                    # Unzip the file
                    with zipfile.ZipFile(os.path.join(data_folder, file_name), 'r') as zip_ref:
                        zip_ref.extractall(data_folder)
                        print(f"📂 Unzipped: {file_name}")

        

# Global dictionaries to store datasets
FC_dataset = {}  # Stores Functional Connectivity (FC) data
PSD_dataset = {}  # Stores EEG PSD data


def load_datasets():
    """Loads and processes datasets, removing unwanted files."""
    global FC_dataset, PSD_dataset  # Ensure global access

    print("🔄 Loading datasets...")

    # ✅ Step 1: Clean up unwanted FC files
    try:
        for root, _, files in os.walk(FC_DATA_PATH):
            for file in files:
                if "_605-" in file:  # Delete files containing '_605-'
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {file_path}")
    except Exception as e:
        print(f"❌ Error while deleting files: {e}")

    # ✅ Step 2: Load Functional Connectivity (FC) datasets
    try:
        for FC_name in os.listdir(FC_DATA_PATH):
            basepath = os.path.join(FC_DATA_PATH, FC_name)
            FC_dataset[FC_name] = get_files_by_class(basepath)
            print(f"📂 Loaded FC dataset: {FC_name}")
    except Exception as e:
        print(f"❌ Error loading FC datasets: {e}")

    # ✅ Step 3: Load EEG PSD dataset
    try:
        PSD_dataset = get_files_by_class(PSD_DATA_PATH)
        print("📂 Loaded PSD dataset successfully!")
    except Exception as e:
        print(f"❌ Error loading PSD dataset: {e}")

    print("✅ Dataset loading complete!")
    
    return FC_dataset, PSD_dataset

# Ensure it only runs when executed directly
if __name__ == "__main__":
    load_datasets()
