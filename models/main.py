# Run the webcam and actually move the mouse with the eye
import cv2
import numpy as np
import dlib
import PIL
from PIL import Image
import imutils
from imutils import face_utils
import torchvision.transforms as transforms
import torch
from model import ConvNet
import pyautogui


cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../classification_data/shape_predictor_68_face_landmarks.dat')

# load the model
model = ConvNet()
model = torch.load('./model.pickle')
model.eval()

to_tensor = transforms.ToTensor()
resize = transforms.Resize((140, 250))

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

        cv2.imshow("ROI", roi)
       
        # esc to cap img
        # if cv2.waitKey(1) == 27: 
        cv2.imwrite('./test.png', roi)
        img = Image.open('./test.png')
        # img = Image.fromarray(roi)
        img = resize(img)
        img = to_tensor(img)
        img = img.reshape(1, 3, 140, 250)

        # run img thru model
        with torch.no_grad():
            output = model.forward(img)
            print("\n", output)
            _, predicted = torch.max(output.data, 1)

            print("PREDICTED: ", predicted)

            if predicted == 0:
            	print("Left")
            	pyautogui.moveRel(-10, 0)

            elif predicted == 1:
            	print("Up")
            	pyautogui.moveRel(0, -10)

            elif predicted == 2:
            	print("Right")
            	pyautogui.moveRel(10, 0)

            elif predicted == 3:
            	pyautogui.moveRel(0, 10)
            	print("Down")

            elif predicted == 4:
            	print("Center")

    cv2.imshow('Image', frame)

cap.release()
cv2.destroyAllWindows()

