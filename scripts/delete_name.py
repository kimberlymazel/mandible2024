import os

def delete_by_name(directory, suffix):
    if not os.path.isdir(directory):
        print("Directory not found.")
        return
    
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            file_path = os.path.join(directory, filename)
            try:
                # Attempt to delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")

# Example usage
directory_path = "dataset/augmented/masks"
suffix_to_delete = "_hflip.png"
delete_by_name(directory_path, suffix_to_delete)
