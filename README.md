# mandible 2024
## Directory
### compare
Folder to store results of different models for comparation

### dataset
Folder containing different datasets used for the models
- annotations & annotations2: Files from Ms. Merve
- box
  - box_original: Original box files from link
  - box_CLAHE: Box images with CLAHE applied
- merged: Merged dataset of [m3-ysqmd](https://universe.roboflow.com/noneed/m3-ysqmd) and [Mendeley](https://data.mendeley.com/datasets/hxt48yk462/1)
- raw: First example of data
- segmentation: Segmented files of annotations2
- xray_panoramic_mandible: Dataset from [Mendeley](https://data.mendeley.com/datasets/hxt48yk462/1)

### models
Folder containing PyTorch files of the trained models 
- clahe_8vs.pt: Model built on a CLAHE datset
- roboflow_8vs.pt: Most recent model built on mixed dataset with roboflow
- yolov8s.pt: First model based on xray_panoramic_mandible

### scripts
Automate processing

### src
Source code 

### test
Test files

### yolo_predictions
Folder containing results of the models based on the box file
