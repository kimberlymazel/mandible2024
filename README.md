# mandible 2023
## Directory
### compare
Folder to store results of different models for comparation

### dataset
Folder containing different datasets used for the models
- annotations: Files from Ms. Merve
- box: Files from the box link
- box_CLAHE: Box images with CLAHE applied
- box_jpg: Box images converted from .tiff to .jpg
- roboflow: Dataset from roboflow (mandible and maxilla)
- roboflow_filtered: Dataset with filtered classes (only mandible)

### model_src
Folder containing the .ipynb files to run the models

### models
Folder containing PyTorch files of the trained models 

### src
Source code other than model running

### yolo_predictions
Folder containing results of the models based on the box file