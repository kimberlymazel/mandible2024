import os

def count_file(dir):
    
    file_count = 0

    for filename in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, filename)):
            file_count += 1

    print("Total number of files in the directory:", file_count)

if __name__ == "__main__":
    directory1 = "forvin/measure"
    directory2 = "forvin/fcn"

    count_file(directory1)
    count_file(directory2)