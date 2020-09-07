import cv2
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox
import glob
from tkinter import filedialog
from tkinter import *
import os
import urllib.request as urlreq
#author hilal yavuz
#adding mustache to images

def openfile():
    global filename
    filename = filedialog.askopenfilename(title="Open file")
    root.destroy()
root = Tk()
root.geometry("200x200")
filename = "" # global variable
button1 = Button(root, text='Browse', command=openfile,activebackground = "red",height=50,width=50).pack()
root.mainloop()

haar_url = "https://github.com/hilalalyavuz/adding_mustache/blob/master/haarcascade_frontalface_alt2.xml"
haar_model = "haarcascade_frontalface_alt2.xml"
haar_file = os.path.join(haar_url, haar_model)
if(haar_model not in os.listdir(os.curdir)):
    urlreq.urlretrieve(haar_file, haar_model)

lbf_url = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
LBFmodel = "lbfmodel.yaml"
lbf_file = os.path.join(lbf_url, LBFmodel)
if(LBFmodel not in os.listdir(os.curdir)):
    urlreq.urlretrieve(lbf_file, LBFmodel)

#detect faces
face_detector = cv2.CascadeClassifier(haar_model)
img = cv2.imread(filename)
imag_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
width_original = imag_gray.shape[1]
height_original = imag_gray.shape[0]
faces = face_detector.detectMultiScale(imag_gray)

#starting landmark on faces
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)
_, landmarks = landmark_detector.fit(imag_gray, faces)
x_coords = []
y_coords = []
for landmark in landmarks:
    for x,y in landmark[0]:
        x_coords.append(x)
        y_coords.append(y)

#adding mustache to faces
mustache_url = r"https://github.com/hilalalyavuz/adding_mustache/blob/master/mustache.png"
mustache_model = "mustache.png"
mustache_file = os.path.join(mustache_url, mustache_model)
if(mustache_model not in os.listdir(os.curdir)):
    urlreq.urlretrieve(mustache_file, mustache_model)
imgMustache = cv2.imread(mustache_model, -1)
orig_mask = imgMustache[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
mustacheWidth = int(abs(3 * (x_coords[31] - x_coords[35])))
mustacheHeight = int(mustacheWidth * origMustacheHeight / origMustacheWidth) - 10
mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
y1 = int(y_coords[33] - (mustacheHeight/2)) + 10
y2 = int(y1 + mustacheHeight)
x1 = int(x_coords[51] - (mustacheWidth/2))
x2 = int(x1 + mustacheWidth)
roi = img[y1:y2, x1:x2]
roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
img[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

#show the image
cv2.imshow('h',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




