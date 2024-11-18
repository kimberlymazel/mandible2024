# mandible 2024

## Directory

### compare

Folder to store results of different models for comparation

### dataset

Folder containing different datasets used for the models

- annotations & annotations2: Files from Dr. Merve
- augmented: Dataset after undergoing augmentation
- box
  - 500_700: 500-700 Pano Fang
  - teeth_nopins: Pano with teeth or no pins
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

#### ./scripts/measure.py

Before running measure.py, please make sure the following python package are installed:

- opencv-python
- scikit-image
- numpy
- imagecodecs (for tiff image)

Run using the following command:

```sh
python ./scripts/measure.py <raw-xray-image relative path> <predicted-mandible-mask relative path>
```

Example:

```sh
python ./scripts/measure.py ./forvin/measure/0103924.tiff ./forvin/unetpp/0103924_prediction.tiff
```

### src

Source code

### test

Test files

## Other Resources

All model files and datasets (train, test, valid) can be accessed in [this drive](https://binusianorg-my.sharepoint.com/personal/kimberly_mazel_binus_ac_id/_layouts/15/guestaccess.aspx?share=Eqe5sPZV8hBDl0rpGt-cZ5kBwBGeHHPr5ujGMjWP21kzlQ&e=ppAGYa)
