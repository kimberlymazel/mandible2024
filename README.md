# mandible 2024
## Directory
### compare
Folder to store results of different models for comparation

### dataset
Folder containing different datasets used for the models
- annotations: Files from Ms. Merve
- box
  - box_original: Original box files from link
  - box_CLAHE: Box images with CLAHE applied
  - box_jpg: Box images converted from .tiff to .jpg
- merged
  - merged1: Merged dataset of [m3-ysqmd](https://universe.roboflow.com/noneed/m3-ysqmd) and [Mendeley](https://data.mendeley.com/datasets/hxt48yk462/1)
  - merged2: Attempted merged dataset of [m3-ysqmd](https://universe.roboflow.com/noneed/m3-ysqmd), [Mendeley](https://data.mendeley.com/datasets/hxt48yk462/1), [lko](https://universe.roboflow.com/mask-wbqqz/lko), [okl](https://universe.roboflow.com/mask-wbqqz/okl), and [yolonihadatsetobject](https://universe.roboflow.com/yolopg/yolonihadatsetobject)
- raw: First example of data
- roboflow
  - lko: Dataset from [lko](https://universe.roboflow.com/mask-wbqqz/lko)
  - lko_filtered: Filtered lko dataset (only mandible)
  - noneed: Dataset from [m3-ysqmd](https://universe.roboflow.com/noneed/m3-ysqmd)
  - noneed_filtered: Filtered noneed dataset (only mandible)
  - okl: Dataset from [okl](https://universe.roboflow.com/mask-wbqqz/okl)
  - okl_filtered: Filtered okl dataset (only mandible)
  - yolonihadatsetobject: Dataset from [yolonihadatsetobject](https://universe.roboflow.com/yolopg/yolonihadatsetobject)
  - yolonihadatsetobject_filtered: Filtered yolonihadatsetobject dataset (only mandible)
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
