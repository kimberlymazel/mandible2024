import os

def check_empty(dir):
    found = False

    for filename in os.listdir(dir):
        if filename.endswith(".txt"):  
            file_path = os.path.join(dir, filename)
            
            if os.path.getsize(file_path) == 0:
                print(f"File {filename} is empty.")
                found = True

    if not found:
        print("No empty files found.")

if __name__ == "__main__":

    directory = "dataset/xray_panoramic_mandible/txt"

    check_empty(directory)
