# mandible 2024
## Directory
### compare
Folder to store results of different models for comparation

### dataset
Folder containing different datasets used for the models
- annotations & annotations2: Files from Ms. Merve
- box
  - 500_700: 500-700 Pano Fang
  - teeth_nopins: Pano with teeth or no pins
- merged: Merged dataset of [m3-ysqmd](https://universe.roboflow.com/noneed/m3-ysqmd) and [Mendeley](https://data.mendeley.com/datasets/hxt48yk462/1)
- raw: First example of data
- segmentation
  - masks: Segmented masks of annotations2
  - txt: Converted masks to txt files
- xray_panoramic_mandible: Dataset from [Mendeley](https://data.mendeley.com/datasets/hxt48yk462/1)

### models
Folder containing PyTorch files of the trained models 
- roboflow_8vs.pt: XPM + noneed model
- xpm_annotate.pt: XPM + annotations2 model
- xpm.pt: XPM model

### scripts
Automate processing

### src
Source code 

### test
Test files
