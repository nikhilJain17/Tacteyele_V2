# Tacteyele_V2

## I. What is this?

Tacteyele is a computer vision program that lets you control your computer without your hands, just with your eyes. My friends and I made Tacteyele_V1 at PennApps XV, a 36 hour hackathon in senior year of high school, where we won 3rd place. 



You can check out Tacteyele_V1 demo here: https://youtu.be/5IFfr-ggy-8?t=1393 

And the (messy) repo here: https://github.com/nikhilJain17/TactEYEle 

The initial vision was to look at a part of the screen and have the mouse move there. However, for Tacteyele_V1, we couldn't get that working and instead used head position. You could click by blinking your eye, and type with voice controls. \


After going through Stanford's CS231n course on ConvNets, I decided to revisit the eye idea. I wrote a script and built a dataset, then trained a classifier on it for different eye poses. Testing accuracy was around 97%. \



Check out Tacteyele_V2 demo here: https://www.youtube.com/watch?v=QS-aNiqF9N8&feature=youtu.be

## II. How's it work?
#### 1. Data Collection and Dataset \
I wrote a script (data_generator.py) that loads the webcam video stream, applies the dlib facial points classifier to segment the image, then crops the eyes and saves the image to a folder. \

This way, you can just hold down the `esc` key and save hundreds of photos very quickly. \

#### 2. Model
The architecture of the model is as follows:
[Conv --> ReLU --> MaxPool] => [Conv --> ReLU --> MaxPool] => Fully Connected \

The optimizer was stochastic gradient descent, with learning rate 0.001, batch size 10, and momentum 0.9. \

This was my first iteration at the classifier. I plan on experimenting with more model architectures, optimizers, transfer learning, etc, in the future.


## III. Installing 

Coming soon!

#### 1. Dependencies
Pytorch \
OpenCV \
PIL \


## IV. What's next?
* Improve framerate!
  * Don't save each frame to run through model
* Smoothen mouse movement!
* Add blink to click
* Add voice commands
* Consider regression based approach
  * Instead of classifying direction (left, right, etc), maybe measuring the angle of the gaze vector
  * Continuous values for direction could give better UX over discrete direction values
* Transfer learning on pretrained model
  * Compare results
  
## V. File structure
```
/classification_data
|  /dataset                                 --> contains training/testing images
|    ...
|  data_generator.py                        --> script to generate training/test images
|  shape_predictor_68_face_landmarks.dat	   --> dlib face classifier, used to segment eye out of image
/experiments                                --> to hold data from future ML experiments
/models
  data_processor.py                         --> defines custom Pytorch dataset class, preprocesses imgs
  model.py                                  --> defines Convnet architecture, only used for class in main.py
  model.ipynb                               --> trains model
  model.pickle                              --> saved weights
  main.py                                   --> runs the model on webcam images, displays a GUI of model outputs, moves mouse
README.md                                   --> see README.md for more info
```
