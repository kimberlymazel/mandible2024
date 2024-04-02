import os

def check_class(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.txt'):
            with open(filepath, 'r') as file:
                lines = file.readlines()
                if len(lines) > 1:
                    print(filename)

if __name__ == "__main__":

    directory = 'dataset/xpm_annotate/labels'
    check_class(directory)
