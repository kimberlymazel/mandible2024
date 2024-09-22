import os
import shutil
import pandas as pd

spreadsheet_path = 'forvin/excelformeasurement-Fang 08.29.xlsx'  
image_column = 'ID #' 
df = pd.read_excel(spreadsheet_path)

image_ids = df[image_column].astype(str).tolist()

source_directories = ['dataset/annotations',
                      'dataset/annotations2', 
                      'dataset/box/500_700', 
                      'dataset/box/teeth_nopins', 
                      'dataset/tinonewpano',
                      'dataset/Need to do measurement']  
destination_directory = 'forvin/original' 

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

files_not_found = set(image_ids) 

for root_dir in source_directories:
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            matching_ids = [image_id for image_id in image_ids if str(image_id) in file]
            if matching_ids:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(destination_directory, file)
                shutil.copy2(source_path, dest_path)
                for image_id in matching_ids:
                    if image_id in files_not_found:
                        files_not_found.remove(image_id)
                print(f"Copied: {file} to {destination_directory}")

if files_not_found:
    print("\nFiles not found:")
    for missing_file in files_not_found:
        print(missing_file)
else:
    print("\nAll files were found and copied!")

print("Script completed!")
