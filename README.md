# CDD-CESM-Dataset
This is a helper repository for the CDD-CESM Mammogram Dataset containing all the tools for pre-processing and segmentation model

Dataset Link: [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8)

Paper Link: [here](https://www.nature.com/articles/s41597-022-01238-0)

<img src="https://user-images.githubusercontent.com/6074821/161619916-46594309-48cd-4853-b56b-7f5c08d2ab8b.png" width="400" height="400">

## Installation & Usage

- pip install -r requirements.txt
- Download dataset
- Put the images inside ```dataset/images```
- split annotations into ```dataset/train_set.csv``` and ```dataset/test_set.csv```
- edit ```configs.py``` to configure the training process 
- run ```train.py``` to train a classification model
- run ```test.py``` to test a classification model
- run ```parse_reports.py``` to parse the full reports and convert them to csv
- run ```clean_images_names.py``` to remove any spaces from the images' names
- run ```parse_reports.py``` to parse the full reports and convert them to csv
- run ```draw_activations.py``` to draw gradcam activations from a trained model
- run ```evaluate_segmentation_model.py``` to evaluate the segmentations from a trained classification model using the method in the paper and save the images
- run ```draw_real_segmentations.py``` to draw the segmentations from the segmentation annotations

<img src="https://user-images.githubusercontent.com/6074821/216845941-d28463aa-6974-4a63-958d-aff44f00f08c.png" width="500" height="400">


## Automatic Segmentation Flow & Example Results

<img src="https://user-images.githubusercontent.com/6074821/216846222-04f48b68-af5d-440c-80bd-7dacab5fc090.png" width="800" height="400">

<img src="https://user-images.githubusercontent.com/6074821/216846289-46fefcf0-5828-4463-85cd-7cc800d94089.png" width="700" height="1000">

## Citation
If you use this dataset, please cite the following:

- Khaled R., Helal M., Alfarghaly O., Mokhtar O., Elkorany A., El Kassas H., Fahmy A. Categorized Digital Database for Low energy and Subtracted Contrast Enhanced Spectral Mammography images [Dataset]. (2021) The Cancer Imaging Archive. DOI:  10.7937/29kw-ae92 

- Khaled, R., Helal, M., Alfarghaly, O., Mokhtar, O., Elkorany, A., El Kassas, H., & Fahmy, A. Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research. (2022) Scientific Data, Volume 9, Issue 1. DOI: 10.1038/s41597-022-01238-0

- Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7



