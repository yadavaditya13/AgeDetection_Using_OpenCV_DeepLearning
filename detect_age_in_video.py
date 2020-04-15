import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_age(frame, faceNet, ageNet, minConf=0.5):

    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    results = []

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > minConf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            index = preds[0].argmax()
            age = AGE_BUCKETS[index]
            ageConfidence = preds[0][index]

            d = {
                "loc": (startX, startY, endX, endY),
                "age": (age, ageConfidence)
            }
            results.append(d)

    return results


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-f", "--face", required=True, help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True, help="path to age detector model directory")
ap.add_argument("-o", "--output", required=True, help="path to output video!...")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

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

# loading video file from disk

print("[INFO] loading video from disk...")
vs = cv2.VideoCapture(args["input"])
writer = None

# calculating its length

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# lets begin our loop for video frames

while(vs.isOpened()):

    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    frame = imutils.resize(frame, width=400)

    results = detect_and_predict_age(frame, faceNet, ageNet, minConf=args["confidence"])

    for r in results:
        text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
        (startX, startY, endX, endY) = r["loc"]

        # lets put text and box on our image

        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)

print("[INFO] Your video file is ready ... Jump to the desired directory of output!...")
vs.release()
writer.release()
cv2.destroyAllWindows()