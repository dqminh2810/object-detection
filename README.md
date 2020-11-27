# object-detection
Mini project allow to detect objects by using R-CNN

# Project structure
.
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
|	├───config.py        // R-CNN model config file
│   ├───iou.py           // IoU method
│   └───nms.py           // NMS method
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
├────results                          // results of objects detection R-CNN model
├────build_dataset.py                 // transform raw-data to dataset used to feed our model
├────detect_object_rcnn_simple.py     // final code used to detect objects on a given image
├────fine_tune_rcnn_simple.py         // build classification model on dataset
├────object_detector_simple.h5        // saved model
├────label_encoder_simple.pickle      // saved label encoder
├────plot_simple.png                  // model evaluation plot

# Utilisation

- Build dataset for object classifier from raw data 
`python buil_dataset.py`

 - Train classification model
 `python fine_tune_rcnn_simpe.py`

 - Detect objects on a given image
 `python detect_object_rcnn_simple.py --image path_to_image`