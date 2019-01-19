# Code to take pictures of user's eyes
# Use this to generate classification dataset
import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils


cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()


# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

counter = 46
while True:
    _, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame

    faces = detector(gray, 1)

    for i, face in enumerate(faces):
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        (x, y, w, h) = cv2.boundingRect(np.array([shape[37:40]]))
        roi = gray[y - 10:y + h + 10, x - 10 : x + w + 10]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        # edge detector
        res = cv2.Canny(roi, 100, 200)

        # show the particular face part
        cv2.imshow("ROI", roi)
       
        # esc to cap img
        if cv2.waitKey(1) == 27: 
            cv2.imwrite('./dataset/validation/up/' + str(counter) + '.png', roi)
            counter += 1

    cv2.imshow('Image', frame)

cap.release()
cv2.destroyAllWindows()

