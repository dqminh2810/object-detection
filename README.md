# object-detection
Mini projet permet de détecter des objets en utilisant R-CNN


# Utilisation

- Build dataset for object classifier from raw data 
`python buil_dataset.py`

 - Train classification model
 `python fine_tune_rcnn_simpe.py`

 - Detect objects on a given image
 `python detect_object_rcnn_simple.py --image path_to_image`