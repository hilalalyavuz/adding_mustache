# adding_mustache_glass
This project adds mustaches and glass to the image what you want to or by using your webcam, directly you. It works on humans images. Firstly, it detects human faces with haarcascade_frontalface_default.xml file and after detecting human faces, it starts to face landmark detection operation. This operation aims to distinguish and mark regions on the human face. To do this operation, I used trained model which is LBF Model. After landmark, it adds mustache and glass to the area where the mustache and glass is on the face. Overall, it shows the image with mustache and glass.

While using this program, you should install haarcascade_frontalface_default.xml and lbfmodel.yaml files but my program already install this files to your computer when you start to run this code. This are for detect and landmark faces. In the beginning of the running time, there is appears a GUI screen and you can select image which you want to operate on it. After choosing the image, the program will do the rest of parts by itself. Or if you run the code with webcam, it starts to show yourself on the screen and adds mustache and glass to your face.

For running this code on your computer, you should add this packages:

import cv2

from tkinter import filedialog

from tkinter import *

from tkinter import messagebox

import glob

from tkinter import filedialog

from tkinter import *

import os

import urllib.request as urlreq
