# mandible 2024
## Directory
### compare
Folder to store results of different models for comparation

### dataset
Folder containing different datasets used for the models
- annotations: Files from Ms. Merve
- box: Files from the box link
- box_CLAHE: Box images with CLAHE applied
- box_jpg: Box images converted from .tiff to .jpg
- merged: Merged dataset of [Roboflow](https://universe.roboflow.com/noneed/m3-ysqmd) and [Mendeley](https://data.mendeley.com/datasets/hxt48yk462/1)
- raw: First example of data
- roboflow: Dataset from [Roboflow](https://universe.roboflow.com/noneed/m3-ysqmd) (mandible and maxilla)
- roboflow_filtered: Dataset with filtered classes (only mandible)
- xray_panoramic_mandible: Dataset from [Mendeley](https://data.mendeley.com/datasets/hxt48yk462/1)

### model_src
Folder containing the .ipynb files to run the models

### models
Folder containing PyTorch files of the trained models 
- clahe_8vs.pt: Model built on a CLAHE datset
- roboflow_8vs.pt: Most recent model built on mixed dataset with roboflow
- yolov8s.pt: First model based on xray_panoramic_mandible

### src
Source code other than model running

### yolo_predictions
Folder containing results of the models based on the box file