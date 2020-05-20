import cv2
import numpy as np
from collections import namedtuple

def playDetectedFrames(path, hsv_min,hsv_max):

    cap = cv2.VideoCapture(path)
    
    frames = []
    count =0

    while True :
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        keypoints = blob_detect(frame,hsv_min, hsv_max )
        iwk = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('asd',iwk)
        k = cv2.waitKey(33)

        if k == 27:
            return
                


def getBlobDetector(exParams):

    key = 'detector_params'

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 256
    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 20000
    params.filterByCircularity = False
    params.minCircularity = 0.1
    params.filterByConvexity = False
    params.minConvexity = 0.5


    ver = cv2.__version__.split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)



def blob_detect(img, params, hsv_min,hsv_max,kernel):


    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, hsv_min, hsv_max)
 
    kernel = np.ones(kernel, np.uint8)

    mask = cv2.dilate(mask,kernel)
    mask = cv2.erode(mask,kernel)

    detector = getBlobDetector(params)
    
    reverseMask = 255 - mask
    keypoints = detector.detect(reverseMask)

    return keypoints


def circleArea(radius):
    return np.pi * radius ** 2


def getRatio(dividend, divisor):
    return dividend / divisor if dividend > 0 else 1 


def calculateMetrics(origKP, cmpKP): 


    blob_tuple = namedtuple("BLOB", ['RATIO', 'OFFSETX', 'OFFSETY', 'SIZERATIO'])
    
    ratio = (len(origKP), len(cmpKP))
    
    offsetx = 0
    offsety = 0
    sizeratio = 0

    oKP = sorted(origKP, key=lambda x: x.size)
    cKP = sorted(cmpKP, key=lambda x: x.size)
    
    if cKP and oKP:
        oKP = oKP[0]
        cKP = cKP[0]

        oPT = oKP.pt
        cPT = cKP.pt
        
        offsetx = abs(oPT[0] - cPT[0])
        offsety = abs(oPT[1] - cPT[1])
        sizeratio = circleArea(cKP.size/2) / circleArea(oKP.size/2)

    return blob_tuple(ratio, offsetx, offsety, sizeratio)


def getBlobMetrics(origImg, cmpImg, params):

    hsv_min, hsv_max, kernel = params

    k1 = blob_detect(origImg, params, hsv_min, hsv_max, kernel)
    k2 = blob_detect(cmpImg, params, hsv_min, hsv_max, kernel)

    return  calculateMetrics(k1, k2)


