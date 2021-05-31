 
import os
from django.urls import path, include
import face_recognition
import cv2
from imutils.video import VideoStream
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array



# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model(r"C:\Users\mkjsr\OneDrive\Desktop\Django_mask_attendance\main_base\mask_detector.model")


def detect_faces(frame,email):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    lable = "Not Verified"

    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    MEDIA_ROOT  = os.path.join(BASE_DIR,'face_dataset')
    loc=(str(MEDIA_ROOT)+'\\'+str(email)+'.jpg')
    face_1_image = face_recognition.load_image_file(loc)
    small_frame_1 = cv2.resize(face_1_image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame_1 = small_frame_1[:, :, ::-1]
    face_1_face_encoding = face_recognition.face_encodings(rgb_small_frame_1)[0]

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        rgb_small_frame = frame[:, :, ::-1]
        face_locations  = face_recognition.face_locations(rgb_small_frame)
        face_encodings  = face_recognition.face_encodings(rgb_small_frame, face_locations)
        if len(face_encodings):

            check = face_recognition.compare_faces(face_1_face_encoding, face_encodings)
            if check[0]:
                    lable = 'Verified'
                    print(lable)

            else :
                    lable = 'Not Verified'
                    print(lable)


    return (locs,lable)

# initialize the camera
def facedect(email):

    cam = VideoStream(src=0).start()   # 0 -> index of camera
    lab = 'Not Verified'
    while True:
        img = cam.read()
        small_frame = imutils.resize(img, width=400)
        # rgb_small_frame = small_frame[:, :, ::-1]
        # face_locations = face_recognition.face_locations(rgb_small_frame)
        # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # check=face_recognition.compare_faces(face_1_face_encoding, face_encodings)

        
        # if check[0]:
        #         label = 'Verified'
        #         print(label)

        # else :
        #         label = 'Verified'
        #         print(label)

                    
        (locs,lable) = detect_faces(small_frame,email)

        # loop over the detected face locations and their corresponding
        # locations
        for box in locs:
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box

            # determine the class label and color we'll use to draw
            # the bounding box and text
            # display the label and bounding box rectangle on the output
            # frame
            color = (0, 255, 0) if lable == "Verified" else (0, 0, 255)

            cv2.putText(small_frame, lable, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(small_frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Frame", small_frame)
        key = cv2.waitKey(2) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            lab = lable
            break
    cv2.destroyAllWindows()
    cam.stop()
    return lab

