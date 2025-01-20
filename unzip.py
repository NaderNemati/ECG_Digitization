import zipfile
import os

# Function to extract a zip file
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

# Define paths for the zip files and the extraction directory
zip_files = [
    "/scratch/project_2010663/ECG_digitalization/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip",
    "/scratch/project_2010663/ECG_digitalization/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
]
extract_to_directory = "/scratch/project_2010663/ECG_digitalization"

# Ensure the extraction directory exists
os.makedirs(extract_to_directory, exist_ok=True)

# Extract each zip file
for zip_file in zip_files:
    extract_zip(zip_file, extract_to_directory)
