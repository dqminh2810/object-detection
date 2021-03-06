# Brings all the pieces together to perform rudimentary R-CNN object detection, 
# the key components being Selective Search and classification 
# - note that this script does not accomplish true end-to-end R-CNN object detection by means of a 
# model with a built-in Selective Search region proposal portion of the network)

# import the necessary packages
from pyimagesearch.nms import non_max_suppression
from pyimagesearch import config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())


####################################
### LOAD PREPAREDMODEL AND LABEL ###
####################################
# load the our fine-tuned model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

# load the input image from disk
image = cv2.imread(args["image"])
image = imutils.resize(image, width=1000)


############################
### GET PROPOSAL REGIONS ###
############################
# run selective search on the image to generate bounding box proposal regions
print("[INFO] running selective search...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

# initialize the list of region proposals that we'll be classifying
# along with their associated bounding boxes
proposals = []
boxes = []

# loop over the region proposal bounding box coordinates generated by
# running selective search
for (x, y, w, h) in rects[:config.MAX_PROPOSALS]:
    # extract the region from the input image, convert it from BGR to
    # RGB channel ordering, and then resize it to the required input
	# dimensions of our trained CNN
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)

    # further preprocess by the ROI
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    # update our proposals and bounding boxes lists
    proposals.append(roi)
    boxes.append((x, y, x + w, y + h))

# convert the proposals and bounding boxes into NumPy arrays
proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")
print("[INFO] proposal shape: {}".format(proposals.shape))
print("[INFO] boxes shape: {}".format(boxes.shape))


##################################################
### APPLY CLASSIFIER MODEL ON PROPOSAL REGIONS ###
##################################################
# classify each of the proposal ROIs using fine-tuned model
print("[INFO] classifying proposals...")
proba = model.predict(proposals)
# find the index of all predictions that are positive for the
# classes
print("[INFO] applying NMS...")
labels = lb.classes_[np.argmax(proba, axis=1)]

idxs_fork = np.where(labels == "fork")[0]
idxs_knife = np.where(labels == "knife")[0]
idxs_plate = np.where(labels == "plate")[0]


# use the indexes to extract all bounding boxes and associated class
# label probabilities associated with the classes
boxes_fork = boxes[idxs_fork]
proba_fork = proba[idxs_fork][:, 1]

boxes_knife = boxes[idxs_knife]
proba_knife = proba[idxs_knife][:, 1]
boxes_plate = boxes[idxs_plate]
proba_plate = proba[idxs_plate][:, 1]

# further filter indexes by enforcing a minimum prediction
# probability be met

if(len(proba_fork) > 0):
    idxs_fork = np.where((proba_fork >= config.MIN_PROBA) | (proba_fork == np.amax(proba_fork)))
    boxes_fork = boxes_fork[idxs_fork]
    proba_fork = proba_fork[idxs_fork]
if(len(proba_knife) > 0):
    idxs_knife = np.where((proba_knife >= config.MIN_PROBA) | (proba_knife == np.amax(proba_knife)))
    boxes_knife = boxes_knife[idxs_knife]
    proba_knife = proba_knife[idxs_knife]
if(len(proba_plate) > 0):
    idxs_plate = np.where((proba_plate >= config.MIN_PROBA) | (proba_plate == np.amax(proba_plate)))
    boxes_plate = boxes_plate[idxs_plate]
    proba_plate = proba_plate[idxs_plate]


print(proba_fork)
print(proba_knife)
print(proba_plate)


#########################################################
### SHOW OBJECT DETECTION RESULTS BEFORE & AFTER NMS  ###
#########################################################
# clone the original image so that we can draw on it
clone = image.copy()

# loop over the bounding boxes and associated probabilities
for (box, prob) in zip(boxes_fork, proba_fork):
    # draw the bounding box, label, and probability on the image
    (startX, startY, endX, endY) = box
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text = "Fork: {:.2f}%".format(prob * 100)
    cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

for (box, prob) in zip(boxes_knife, proba_knife):
    # draw the bounding box, label, and probability on the image
    (startX, startY, endX, endY) = box
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text = "Knife: {:.2f}%".format(prob * 100)
    cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

for (box, prob) in zip(boxes_plate, proba_plate):
    # draw the bounding box, label, and probability on the image
    (startX, startY, endX, endY) = box
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text = "Plate: {:.2f}%".format(prob * 100)
    cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# show the output after *before* running NMS
cv2.imshow("Before NMS", clone)

# run non-maxima suppression on the bounding boxes
boxIdxs_fork = non_max_suppression(boxes_fork, proba_fork)
boxIdxs_knife = non_max_suppression(boxes_knife, proba_knife)
boxIdxs_plate = non_max_suppression(boxes_plate, proba_plate)


# loop over the bounding box indexes
for i in boxIdxs_fork:
    # draw the bounding box, label, and probability on the image
    (startX, startY, endX, endY) = boxes_fork[i]
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text= "Fork: {:.2f}%".format(proba_fork[i] * 100)

    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

for i in boxIdxs_knife:
    # draw the bounding box, label, and probability on the image
    (startX, startY, endX, endY) = boxes_knife[i]
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text= "Knife: {:.2f}%".format(proba_knife[i] * 100)
    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

for i in boxIdxs_plate:
    # draw the bounding box, label, and probability on the image
    (startX, startY, endX, endY) = boxes_plate[i]
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text= "Plate: {:.2f}%".format(proba_plate[i] * 100)

    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


# show the output image *after* running NMS
cv2.imshow("After NMS", image)
cv2.waitKey(0)
