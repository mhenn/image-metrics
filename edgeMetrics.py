import numpy as np
import cv2



def getWhitePixel(img):
    c = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y] > 0:
                c+=1
    return c


def getWhitePercentage(img):    
    return getWhitePixel(img) / (img.shape[0]*img.shape[1])


def percentageComparison(img, cmpImg):
    
    p1 = getWhitePercentage(img) * 100
    p2 = getWhitePercentage(cmpImg) * 100
    return (p2 * 100) / p1



def blur(img, kernel, rounds): 
    im = img.copy()
    for i in range(rounds):
        im = cv2.GaussianBlur(im,kernel,cv2.BORDER_DEFAULT);
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


def prepareCalc(origImg, cmpImg, kernel, rounds):
    
    oim = blur(origImg, kernel, rounds)
    cim = blur(cmpImg, kernel, rounds)
    oPixel = getWhitePixel(oim) 
    return (oim,cim, oPixel)


def getEdgeMSE(params):

    oim, cim, oPixel = params
    dst = np.square(np.subtract(oim, cim))
    dPixel = getWhitePixel(dst)
    return dPixel / oPixel


def getMultiplyStuff(params):
    oim, cim, oPixel = params
    dst = np.multiply(oim, cim)
    dPixel = getWhitePixel(dst)
    return dPixel / oPixel


def getEdgeMetrics(origImg, cmpImg, params):

    edgeMetrics = {}
    i0, i1 = origImg.copy(), cmpImg.copy()

    if 'gray' in list(params.keys()):
        i0 = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
        i1 = cv2.cvtColor( i1, cv2.COLOR_BGR2GRAY)
    
    img = cv2.Canny(i0, 50, 200)
    img2 = cv2.Canny(i1, 50, 200)
        
    calcParams = prepareCalc(img,img2, params['kernel'], params['blur_rounds'])

    edgeMetrics['MSE'] = getEdgeMSE(calcParams)
    edgeMetrics['MULTIPLYSTUFF'] = getMultiplyStuff(calcParams)
    return edgeMetrics
