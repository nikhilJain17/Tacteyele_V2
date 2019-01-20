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
import autopy
from math import exp

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

        if cv2.waitKey(1) == 27:
        	pass


        cv2.imwrite('./test.png', roi)
        img = Image.open('./test.png')
        # img = Image.fromarray(roi)
        img = resize(img)
        img = to_tensor(img)
        img = img.reshape(1, 3, 140, 250)

        results = []

        # run img thru model
        with torch.no_grad():
            output = model.forward(img)
            print("\n", output.data)
            value, predicted = torch.max(output.data, 1)

            results_dict = {0: "Left", 1: "Up", 2: "Right", 3: "Down", 4: "Center"}

            print("PREDICTED: ", predicted, results_dict[int(predicted)])

            # movement = value * 10
            mouse_x, mouse_y = autopy.mouse.location()
            movement = 20


            # resultStr = str(output.data)
            exp_probs = [exp(prob) for prob in output.data[0]]
            probs = ['{0:f}'.format(prob / sum(exp_probs)) for prob in exp_probs]
            results = [results_dict[index] + ": " + str(prob) + "\n" for index, prob in enumerate(probs)]


            if predicted == 0:
            	# resultStr = "Left"
            	autopy.mouse.smooth_move(mouse_x - movement - movement / 2, mouse_y)

            elif predicted == 1:
            	# resultStr = "Up"
            	autopy.mouse.smooth_move(mouse_x, mouse_y - movement)

            elif predicted == 2:
            	# resultStr = "Right"
            	autopy.mouse.smooth_move(mouse_x + movement + movement / 2, mouse_y)

            elif predicted == 3:
            	autopy.mouse.smooth_move(mouse_x, mouse_y + movement)
            	# resultStr = "Down"

            elif predicted == 4:
            	# resultStr = "Center"
            	pass


        for i, string in enumerate(results):
            cv2.putText(gray, string, (100, (i + 1) * 100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 1)

    cv2.imshow('Image', gray)

cap.release()
cv2.destroyAllWindows()

