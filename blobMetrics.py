import cv2
import numpy as np

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


def calculateMetrics(origKP, cmpKP, metric): 

    metric.BLOBRATIO.append(getRatio(len(origKP),len(cmpKP)))

    oKP = sorted(origKP, key=lambda x: x.size)
    cKP = sorted(cmpKP, key=lambda x: x.size)
    if cKP and cKP:
        oKP = oKP[0]
        cKP = cKP[0]

        oPT = oKP.pt
        cPT = cKP.pt

        metric.OFFSETX.append( abs(oPT[0] - cPT[0]))
        metric.OFFSETY.append( abs(oPT[1] - cPT[1]))
        metric.SIZERATIO.append( circleArea(cKP.size/2) / circleArea(oKP.size/2))
    else:
        metric.OFFSETX.append( 0)
        metric.OFFSETY.append( 0 )
        metric.SIZERATIO.append( 0)

    return metric 


def getBlobMetrics(origImg, cmpImg, params, metric):

    hsv_min = params.blob_hsv_min
    hsv_max = params.blob_hsv_max
    kernel = params.blob_blur_kernel

    k1 = blob_detect(origImg, params, hsv_min, hsv_max, kernel)
    k2 = blob_detect(cmpImg, params, hsv_min, hsv_max, kernel)

    return  calculateMetrics(k1, k2,metric)


