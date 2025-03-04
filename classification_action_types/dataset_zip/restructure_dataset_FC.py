import os
import shutil
import zipfile

# Ensure you are in the 'classification_action_types' subproject folder

# Define the folder containing the dataset zip file
dataset_zip_folder = os.path.join(os.getcwd(), 'dataset_zip')

# Unzip the dataset file if it exists
if os.path.exists(dataset_zip_folder):
    file_name = 'connectivity_data_by_action_archive.zip'
    # Unzip the file
    with zipfile.ZipFile(os.path.join(dataset_zip_folder, file_name), 'r') as zip_ref:
        zip_ref.extractall(dataset_zip_folder)
        print(f"📂 Unzipped: {file_name}")

# Define the folder containing the unzipped dataset
dataset_folder = os.path.join(dataset_zip_folder, 'connectivity_data_by_action')

# Iterate over each FC_name directory in the dataset folder
for FC_name in os.listdir(dataset_folder):
    print(FC_name)
    FC_path = os.path.join(dataset_folder, FC_name)
    
    # Iterate over each folder within the FC_name directory
    for folder in os.listdir(FC_path):
        
        # Check if the folder name ends with a digit from 0 to 4 and does not start with 'a' or '5'
        for i in range(5):
            if (folder[-1] == str(i)) and (folder[0] != 'a') and (folder[0] != '5'):
                print('moving', folder)
                target_folder = 'action_' + str(i)
                target_path = os.path.join(FC_path, target_folder)
                source_path = os.path.join(FC_path, folder)
                
                # Create the target folder if it doesn't exist
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                
                # Move files from the source folder to the target folder
                for file_name in os.listdir(source_path):
                    shutil.move(os.path.join(source_path, file_name), os.path.join(target_path, file_name))

                # Remove the source folder after moving its contents
                shutil.rmtree(source_path)

            # Check if the folder name starts with '5' and remove it
            elif folder[0] == '5':
                source_path = os.path.join(FC_path, folder)
                if os.path.exists(source_path):
                    shutil.rmtree(source_path)

# Create a temporary directory to hold the restructured dataset
temp_dir = os.path.join(dataset_zip_folder, 'connectivity_data_by_action_temp')
os.makedirs(temp_dir, exist_ok=True)

# Move the restructured dataset folder into the temporary directory
shutil.move(dataset_folder, os.path.join(temp_dir, 'connectivity_data_by_action'))

# Zip the temporary directory
output_zip_file = os.path.join(dataset_zip_folder, 'connectivity_data_by_action.zip')
shutil.make_archive(output_zip_file.replace('.zip', ''), 'zip', temp_dir)
print(f"📦 Zipped: {output_zip_file}")

# Clean up the temporary directory
shutil.rmtree(temp_dir)

# Remove the unzipped dataset folder after zipping
shutil.rmtree(dataset_folder)