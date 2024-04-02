import os
import pandas as pd

def check_pair(dir1, dir2):

    d1 = sorted(os.listdir(dir1))
    d2 = sorted(os.listdir(dir2))

    max_length = max(len(d1), len(d2))

    df = pd.DataFrame({'Directory 1': d1 + [''] * (max_length - len(d1)),
                       'Directory 2': d2 + [''] * (max_length - len(d2))})

    csv_file_path = "compare.csv"
    df.to_csv(csv_file_path, index=False)

    print(f"DataFrame exported to CSV file: {csv_file_path}")

if __name__ == "__main__":

    directory1 = "dataset/xpm_annotate/yolo/test/images"
    directory2 = "dataset/xpm_annotate/images"

    check_pair(directory1, directory2)