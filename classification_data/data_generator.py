# Code to take pictures of user's eyes
# Use this to generate classification dataset
import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils


cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

counter = 0
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
            cv2.imwrite('./dataset/center/' + str(counter) + '.png', roi)
            counter += 1




        # # top left: (x1, y1), bottom right: (x2, y2)
        # x1, _ = shape[37]
        # _, y1 = shape[38]
        # x2, _ = shape[46]
        # _, y2 = shape[47]

        # cv2.rectangle(gray,(x2,y2),(x2 - x1, y2 - y1),(255,0,0),4)



    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # display_img = frame

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)

    #     roi = gray[y:y+h, x:x+w]
    #     eyes = eye_cascade.detectMultiScale(roi)

    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 4)

    #     # display_img = roi

    cv2.imshow('Image', frame)
    


cap.release()
cv2.destroyAllWindows()

