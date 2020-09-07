# adding_mustache
This project adds mustaches to the image what you want to. It works on humans images. Firstly, it detects human faces with haarcascade_frontalface_alt2.xml file and after detecting human faces, it starts to face landmark detection operation. This operation aims to distinguish and mark regions on the human face. To do this operation, I used trained model which is LBF Model. After landmark, it adds mustache to the area where the mustache is on the face. Overall, it shows the image with mustache.

While using this program, you should install files which are I uploaded under this repo. This are for detect and landmark faces. In the beginning of the running time, there is appears a GUI screen and you can select image which you want to operate on it. After choosing the image, the program will do the rest of parts by itself.

For running this code on your computer, you should add this packages:

import cv2

from tkinter import filedialog

from tkinter import *

from tkinter import messagebox

import glob

from tkinter import filedialog

from tkinter import *
