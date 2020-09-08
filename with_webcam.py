import cv2
import glob
import os
import urllib.request as urlreq
#author hilal yavuz
#adding mustache and glass to images

haar_url = "https://github.com/hilalalyavuz/adding_mustache/blob/master/haarcascade_frontalface_default.xml"
haar_file = "haarcascade_frontalface_default.xml"

if(haar_file not in os.listdir(os.curdir)):
    urlreq.urlretrieve(haar_url, haar_file)

lbf_url = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
LBFmodel = "lbfmodel.yaml"
if(LBFmodel not in os.listdir(os.curdir)):
    urlreq.urlretrieve(lbf_url, LBFmodel)

#detect faces
face_detector = cv2.CascadeClassifier(haar_file)

#load trained landmark model
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

#start webcam
cap = cv2.VideoCapture(0)

#adding mustache to faces
mustache_url = "https://raw.githubusercontent.com/hilalalyavuz/adding_mustache/master/mustache.png"
mustache_file = "mustache.png"
if(mustache_file not in os.listdir(os.curdir)):
    urlreq.urlretrieve(mustache_url, mustache_file)
imgMustache = cv2.imread(mustache_file,-1)
imgMustache = cv2.imread(mustache_file,-1)
orig_mask = imgMustache[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

#adding glass to faces
glass_url = "https://raw.githubusercontent.com/hilalalyavuz/adding_mustache_glass/master/glasses.png"
glass_file = "glasses.png"
if(glass_file not in os.listdir(os.curdir)):
    urlreq.urlretrieve(glass_url, glass_file)
imgGlass = cv2.imread(glass_file,-1)
orig_mask_g = imgGlass[:,:,3]
orig_mask_inv_g = cv2.bitwise_not(orig_mask_g)
imgGlass = imgGlass[:,:,0:3]
origGlassHeight, origGlassWidth = imgGlass.shape[:2]

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)
    _, landmarks = landmark_detector.fit(frame, faces)
    x_coords = []
    y_coords = []
    for landmark in landmarks:
        for x,y in landmark[0]:
            x_coords.append(x)
            y_coords.append(y)

    mustacheWidth = int(abs(3 * (x_coords[31] - x_coords[35])))
    mustacheHeight = int(mustacheWidth * origMustacheHeight / origMustacheWidth) - 10
    mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
    y1 = int(y_coords[33] - (mustacheHeight/2)) + 10
    y2 = int(y1 + mustacheHeight)
    x1 = int(x_coords[51] - (mustacheWidth/2))
    x2 = int(x1 + mustacheWidth)
    roi = frame[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
    frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)


    glassWidth = int(abs(x_coords[16] - x_coords[1]))
    glassHeight = int(glassWidth * origGlassHeight / origGlassWidth)
    glass = cv2.resize(imgGlass, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
    masks = cv2.resize(orig_mask_g, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
    masks_inv = cv2.resize(orig_mask_inv_g, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
    y1 = int(y_coords[24])
    y2 = int(y1 + glassHeight)
    x1 = int(x_coords[27] - (glassWidth/2))
    x2 = int(x1 + glassWidth)
    roi1 = frame[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi1,roi1,mask = masks_inv)
    roi_fg = cv2.bitwise_and(glass,glass,mask = masks)
    frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

    #show the image
    cv2.imshow('hilal',frame)
    keyCode = cv2.waitKey(2)

    if cv2.getWindowProperty('hilal', cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
cap.release()







