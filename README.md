# object-detection
Mini project allow to detect objects by using R-CNN

# Project structure
```
├───dataset              // dataset used to feed our model
│   ├───testing
│   └───training
│       │ 
│       └───simple
│           ├───fork     [548 entries]
│           ├───knife    [806 entries]
│           ├───no_fork  [1418 entries] 
│           ├───no_knife [1067 entries]
│           ├───no_plate [1026 entries]
│           └───plate    [964 entries]
├───pyimagesearch
│       ├───noconfig.py        // R-CNN model config file
│       ├───noiou.py           // IoU method
│       └───plnms.py           // NMS method
├───raw-data			 // raw data downloaded form open images base
│   ├───fork
│   │   ├───images       [411 entries]
│   │   └───pascal       [411 entries]
│   ├───knife
│   │   ├───images       [596 entries]
│   │   └───pascal       [596 entries]
│   └───plate
│       ├───images       [732 entries]
│       └───pascal       [732 entries]
├────results
├────build_dataset.py                 // transform raw-data to dataset used to feed our model
├────detect_object_rcnn_simple.py     // final code used to detect objects on a given image
├────fine_tune_rcnn_simple.py         // build classification model on dataset
├────object_detector_simple.h5        // saved model
├────label_encoder_simple.pickle      // saved label encoder
├────plot_simple.png                  // model evaluation plot
```
# Technologies used

- Python v3.7.2
- pip v9.0.3

# Python libraries used

- OpenCV Python
- bs4
- Tensorflow
- matplotlib
- argparse
- numpy
- imutils
- pickle
- os

# Utilization

- Build dataset from raw data in order to feed object classifier 

`python buil_dataset.py`

 - Train classification model

 `python fine_tune_rcnn_simpe.py`

 - Detect objects on a given image

 `python detect_object_rcnn_simple.py --image path_to_image`

__ **Note**  __
 `After getting the src code, you can run the object detection program directly by executing the command of detect objects above. The saved classifier model as oject_detector_simple.h5 will be used in the case`

*Example*
`python detect_object_rcnn_simple.py --image dataset/testing/knife.jpg`