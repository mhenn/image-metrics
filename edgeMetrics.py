import numpy as np
import cv2
import matplotlib.pyplot as plt

def getWhitePixel(img):
    return np.where(img > 0, 1,0).sum()

def getWhitePercentage(img):    
    return getWhitePixel(img) / (img.shape[0]*img.shape[1])

def percentageComparison(img, cmpImg):
    p1 = getWhitePercentage(img) * 100
    p2 = getWhitePercentage(cmpImg) * 100
    return (p2 * 100) / p1

def blur(img, kernel, rounds): 
    im = img.copy()
    for i in range(rounds):
        im = cv2.GaussianBlur(im,kernel,cv2.BORDER_DEFAULT)
    return im


def naiveOverlayCmp(img,cmpImg, kernel, gaussRounds):

    height, width, channels  = img.shape
    i1 = img.copy()
    i2 = cmpImg.copy()
        
    for i in range(gaussRounds):
        i1 = cv2.GaussianBlur(i1,kernel,cv2.BORDER_DEFAULT);
        i2 = cv2.GaussianBlur(i2,kernel,cv2.BORDER_DEFAULT);

    c = 0
    dst = np.zeros(img.shape)
    

    for x in range(width):
        for y in range(height):
            if i1[y,x].all() and i2[y,x].all() > 0 :
                dst[y,x] = 1
    return dst


def getEdgeMSE(img1, img2, kernel, rounds):

    oim = blur(img1,kernel,rounds)
    cim = blur(img2,kernel,rounds)
    dst = np.square(np.subtract(oim, cim))
    s_sum = dst.sum()
    return s_sum 


def getEdgeResponse(img1, img2, kernel,rounds):
    totalWPCount = getWhitePixel(img1) + getWhitePixel(img2) 
    
    img1 = blur(img1,kernel, rounds)
    img2 = blur(img2,kernel, rounds)
    
    img1 = cv2.GaussianBlur(img1, kernel, cv2.BORDER_DEFAULT)
    img2 = cv2.GaussianBlur(img2, kernel, cv2.BORDER_DEFAULT)

    norm1 = np.divide(img1, 255.)
    norm2 = np.divide(img2, 255.)
    corr = np.multiply(norm1, norm2)
    return (2 * corr.sum()) / (totalWPCount + 1)


def getEdgeMetrics(origImg, cmpImg, params):

    kernel = params.edge_blur_kernel
    rounds = params.edge_blur_rounds
    t1 = params.edge_canny_thresh1
    t2 = params.edge_canny_thresh2

    edgeMetrics = {}
    i0, i1 = origImg.copy(), cmpImg.copy()

    if 'gray' in list(params.__dict__):
        i0 = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
        i1 = cv2.cvtColor( i1, cv2.COLOR_BGR2GRAY)
    
    img = cv2.Canny(i0, t1, t2)
    img2 = cv2.Canny(i1, t1, t2)        

    edgeMetrics['EDGE-MSE'] = getEdgeMSE(i0,i1,kernel,rounds)
    edgeMetrics['EDGE-RESPONSE'] = getEdgeResponse(i0,i1,kernel,rounds)
    return edgeMetrics
