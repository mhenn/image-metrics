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

    if key in list(exParams.keys()):
        params = exParams[key]
    else:
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



def blob_detect(img, params):

    hsv_min, hsv_max = params['hsv_min'], params['hsv_max']

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, hsv_min, hsv_max)
 
    kernel = np.ones(params['kernel'], np.uint8)

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

    blobMetrics = {}
    blobMetrics['blobratio'] = getRatio(len(origKP),len(cmpKP))

    oKP = sorted(origKP, key=lambda x: x.size)
    cKP = sorted(cmpKP, key=lambda x: x.size)
    
    oKP = oKP[0]
    cKP = cKP[0]

    oPT = oKP.pt
    cPT = cKP.pt

    blobMetrics['offsetX'] = abs(oPT[0] - cPT[0])
    blobMetrics['offsetY'] = abs(oPT[1] - cPT[1])
    blobMetrics['sizeRatio'] = circleArea(cKP.size/2) / circleArea(oKP.size/2)
    return blobMetrics 


def getBlobMetrics(origImg, cmpImg, params):

    k1 = blob_detect(origImg, params)
    k2 = blob_detect(cmpImg, params)

    metrics = calculateMetrics(k1, k2)
    return metrics


