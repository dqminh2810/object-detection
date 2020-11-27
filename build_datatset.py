# take raw data and create dataset which will be used to fine-tune a MobileNet V2 model (a pre-trained on the ImageNet dataset) 

# import the necessary packages
from pyimagesearch.iou import compute_iou
from pyimagesearch import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os

####################################
### GET POSITIVE PROPOSAL REGIONS ###
####################################
def getPositiveRois(classe, imagePaths, totalAdding):
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # show a progress report
        print("[INFO] processing image {}/{} to get positive rois of class {}...".format(i + 1, len(imagePaths), classe))
        # extract the filename from the file path and use it to derive
        # the path to the XML annotation file
        filename = imagePath.split(os.path.sep)[-1]
        filename = filename[:filename.rfind(".")]

        annotPath = ""
        if classe == "fork":
            annotPath = os.path.sep.join([config.FORK_ORIG_ANNOTS, "{}.xml".format(filename)])
        elif classe == "knife":
            annotPath = os.path.sep.join([config.KNIFE_ORIG_ANNOTS, "{}.xml".format(filename)])
        elif classe == "plate":
            annotPath = os.path.sep.join([config.PLATE_ORIG_ANNOTS, "{}.xml".format(filename)])

        # load the annotation file, build the soup, and initialize our
        # list of ground-truth bounding boxes
        contents = open(annotPath).read()
        soup = BeautifulSoup(contents, "html.parser")
        gtBoxes = []
        # extract the image dimensions
        w = int(soup.find("width").string)
        h = int(soup.find("height").string)

        # loop over all 'object' elements
        for o in soup.find_all("object"):
            # extract the label and bounding box coordinates
            label = o.find("name").string
            xMin = int(o.find("xmin").string)
            yMin = int(o.find("ymin").string)
            xMax = int(o.find("xmax").string)
            yMax = int(o.find("ymax").string)

            # truncate any bounding box coordinates that may fall
            # outside the boundaries of the image
            xMin = max(0, xMin)
            yMin = max(0, yMin)
            xMax = min(w, xMax)
            yMax = min(h, yMax)

            # update our list of ground-truth bounding boxes
            gtBoxes.append((xMin, yMin, xMax, yMax))

        # load the input image from disk
        image = cv2.imread(imagePath)

        for gtBox in gtBoxes:
            # compute the intersection over union between the two
            # boxes and unpack the ground-truth bounding box
            (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

            # initialize the ROI and output path
            roi = None
            outputPath = None
            (startX, startY, endX, endY) = gtBox

            
            # extract the ROI and then derive the output path to
            # the positive instance
            roi = image[startY:endY, startX:endX]

            if classe == "fork":
                filename = "{}.png".format(totalAdding)
                outputPath = os.path.sep.join([config.SIMPLE_FORK_TRAINING_BASE_PATH, filename])
            elif classe == "knife":
                filename = "{}.png".format(totalAdding)
                outputPath = os.path.sep.join([config.SIMPLE_KNIFE_TRAINING_BASE_PATH, filename])
            elif classe == "plate":
                filename = "{}.png".format(totalAdding)
                outputPath = os.path.sep.join([config.SIMPLE_PLATE_TRAINING_BASE_PATH, filename])

            # increment the positive counters
            totalAdding += 1

            # check to see if both the ROI and output path are valid
            if roi is not None and outputPath is not None:
                # resize the ROI to the input dimensions of the CNN
                # that we'll be fine-tuning, then write the ROI to
                # disk
                roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)

#####################################
### GET NEGATIVE PROPOSAL REGIONS ###
#####################################
def getNegativeRois(classe, imagePaths, totalAdding):
    # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            if i > 7:
                # show a progress report
                print("[INFO] processing image {}/{} to get negative rois of class {} --- total negative features adding .... {}".format(i + 1,len(imagePaths), classe, totalAdding))

                # extract the filename from the file path and use it to derive
                # the path to the XML annotation file
                filename = imagePath.split(os.path.sep)[-1]
                filename = filename[:filename.rfind(".")]
                
                annotPath = ""
                if classe == "fork":
                    annotPath = os.path.sep.join([config.FORK_ORIG_ANNOTS, "{}.xml".format(filename)])
                elif classe == "knife":
                    annotPath = os.path.sep.join([config.KNIFE_ORIG_ANNOTS, "{}.xml".format(filename)])
                elif classe == "plate":
                    annotPath = os.path.sep.join([config.PLATE_ORIG_ANNOTS, "{}.xml".format(filename)])

                # load the annotation file, build the soup, and initialize our
                # list of ground-truth bounding boxes
                contents = open(annotPath).read()
                soup = BeautifulSoup(contents, "html.parser")
                gtBoxes = []

                # extract the image dimensions
                w = int(soup.find("width").string)
                h = int(soup.find("height").string)

                # loop over all 'object' elements
                for o in soup.find_all("object"):
                    # extract the label and bounding box coordinates
                    label = o.find("name").string
                    xMin = int(o.find("xmin").string)
                    yMin = int(o.find("ymin").string)
                    xMax = int(o.find("xmax").string)
                    yMax = int(o.find("ymax").string)

                    # truncate any bounding box coordinates that may fall
                    # outside the boundaries of the image
                    xMin = max(0, xMin)
                    yMin = max(0, yMin)
                    xMax = min(w, xMax)
                    yMax = min(h, yMax)

                    # update our list of ground-truth bounding boxes
                    gtBoxes.append((xMin, yMin, xMax, yMax))

                    # load the input image from disk
                    image = cv2.imread(imagePath)

                    # run selective search on the image and initialize our list of
                    # proposed boxes
                    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                    ss.setBaseImage(image)
                    ss.switchToSelectiveSearchFast()
                    rects = ss.process()
                    proposedRects= []

                    # loop over the rectangles generated by selective search
                    for (x, y, w, h) in rects:
                        # convert our bounding boxes from (x, y, w, h) to (startX,
                        # startY, startX, endY)
                        proposedRects.append((x, y, x + w, y + h))

                    # initialize counters used to count the number of positive and
                    # negative ROIs saved thus far
                    # positiveROIs = 0
                    negativeROIs = 0

                    # loop over the maximum number of region proposals
                for proposedRect in proposedRects[:config.MAX_PROPOSALS_INFER]:
                    # unpack the proposed rectangle bounding box
                    (propStartX, propStartY, propEndX, propEndY) = proposedRect

                    for gtBox in gtBoxes:
                    # compute the intersection over union between the two
                    # boxes and unpack the ground-truth bounding box
                        iou = compute_iou(gtBox, proposedRect)
                        (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox            

                        # initialize the ROI and output path
                        roi = None
                        outputPath = None
                    

                        # POSITIVE
                        '''
                        # check to see if the IOU is greater than 70% *and* that
                        # we have not hit our positive count limit
                        if iou > 0.7 and positiveROIs <= config.MAX_POSITIVE:
                            # extract the ROI and then derive the output path to
                            # the positive instance
                            roi = image[propStartY:propEndY, propStartX:propEndX]
                            filename = "{}.png".format(totalPositive)
                            outputPath = os.path.sep.join([config.POSITVE_PATH,
                                filename])

                            # increment the positive counters
                            positiveROIs += 1
                            totalAdding += 1
                        '''


                        # determine if the proposed bounding box falls *within*
                        # the ground-truth bounding box
                        fullOverlap = propStartX >= gtStartX
                        fullOverlap = fullOverlap and propStartY >= gtStartY
                        fullOverlap = fullOverlap and propEndX <= gtEndX
                        fullOverlap = fullOverlap and propEndY <= gtEndY

                        # NEGATIVE
                        # check to see if there is not full overlap *and* the IoU
                        # is less than 5% *and* we have not hit our negative
                        # count limit
                        if not fullOverlap and iou < 0.05 and \
                            negativeROIs <= config.MAX_NEGATIVE:
                            # extract the ROI and then derive the output path to
                            # the negative instance
                            roi = image[propStartY:propEndY, propStartX:propEndX]

                            if classe == "fork":
                                filename = "{}.png".format(totalAdding)
                                outputPath = os.path.sep.join([config.SIMPLE_NO_FORK_TRAINING_BASE_PATH, filename])
                            elif classe == "knife":
                                filename = "{}.png".format(totalAdding)
                                outputPath = os.path.sep.join([config.SIMPLE_NO_KNIFE_TRAINING_BASE_PATH, filename])
                            elif classe == "plate":
                                filename = "{}.png".format(totalAdding)
                                outputPath = os.path.sep.join([config.SIMPLE_NO_PLATE_TRAINING_BASE_PATH, filename])

                            # increment the negative counters
                            negativeROIs += 1
                            totalAdding += 1

                        # check to see if both the ROI and output path are valid
                        if roi is not None and outputPath is not None:
                            # resize the ROI to the input dimensions of the CNN
                            # that we'll be fine-tuning, then write the ROI to
                            # disk
                            roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite(outputPath, roi)

                        # we take only 500 features of negative class
                        #if totalAdding == 500:
                            #break



###################################
### PREPARE DATASET TO TRAINING ###
###################################
# paths contains training images
img_paths = [config.SIMPLE_FORK_TRAINING_BASE_PATH, 
                config.SIMPLE_KNIFE_TRAINING_BASE_PATH, 
                config.SIMPLE_PLATE_TRAINING_BASE_PATH,
                config.SIMPLE_NO_FORK_TRAINING_BASE_PATH,
                config.SIMPLE_NO_KNIFE_TRAINING_BASE_PATH,
                config.SIMPLE_NO_PLATE_TRAINING_BASE_PATH]

# loop over the output positive and negative directories
for dirPath in img_paths:
    # if the output directory does not exist yet, create it
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

# grab all image paths in the input images directory
fork_imagePaths = list(paths.list_images(config.FORK_ORIG_IMAGES))
knife_imagePaths = list(paths.list_images(config.KNIFE_ORIG_IMAGES))
plate_imagePaths = list(paths.list_images(config.PLATE_ORIG_IMAGES))

# initialize the total number of positive and negative images we have
# saved to disk so far
totalFork_positive = 0
totalKnife_positive = 0
totalPlate_positive = 0
totalFork_negative = 0
totalKnife_negative = 0
totalPlate_negative = 0


# Get Positive rois
getPositiveRois("fork", fork_imagePaths, totalFork_positive)
getPositiveRois("knife", knife_imagePaths, totalKnife_positive)
getPositiveRois("plate", plate_imagePaths, totalPlate_positive)

# Get Negative rois
getNegativeRois("fork", fork_imagePaths, totalFork_negative)
getNegativeRois("knife", knife_imagePaths, totalKnife_negative)
getNegativeRois("plate", plate_imagePaths, totalPlate_negative)

