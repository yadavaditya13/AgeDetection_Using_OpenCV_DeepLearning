import numpy as np
import argparse
import os
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-f", "--face", required=True, help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True, help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# the problem of age detector is considered as a classifier here.
# following is the class label for our pre-trained age detector

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# loading face detector model

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# loading age detector model

print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing face detections in input image...")
faceNet.setInput(blob)
detections = faceNet.forward()

# now we will loop over our detections or simply stated Region of interest

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        #we blob face to get roi
        face = image[startY:endY, startX:endX]
        faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        ageNet.setInput(faceBlob)
        preds = ageNet.forward()

        index = preds[0].argmax()
        age = AGE_BUCKETS[index]
        ageConfidence = preds[0][index]

        text = "{}: {:.2f}%".format(age, ageConfidence * 100)
        print("[INFO] {}".format(text))

        # lets put text and box on our image

        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)