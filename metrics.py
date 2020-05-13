from frames import * 
from edgeMetrics import *
from blobMetrics import *
from parameters import *

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2



def getSSIM(origImg, cmpImg):
    return  ssim(origImg, cmpImg,data_range=cmpImg.max() - cmpImg.min(), multichannel=True)

def getMSSSIM(origImg, cmpImg, weights):
    width, height, channels = origImg.shape
    msssim = 0
    
    if origImg.shape != cmpImg.shape:
        return -1

    i0 = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
    i1 = cv2.cvtColor(cmpImg, cv2.COLOR_BGR2GRAY)


    for weight in weights:
        width = int((width /2))
        height = int((height / 2))

        i0 = cv2.resize(i0, (width,height))
        i1 = cv2.resize(i1, (width,height))
        
        msssim += getSSIM(i0,i1 ) * weight

    return msssim


def printEdgeMetrics():

    edgeParams = {}
    edgeParams['kernel'] = (7,7)
    edgeParams['blur_rounds'] = 1
    frames = getFrames('videos/my_video-2.mkv', 1)
    compareFrames = getFrames('videos/newest_test.mp4', 1)
    
    #dst = naiveOverlayCmp(frames[0], compareFrames[0] , (3,3), 1)
    print(getEdgeMetrics(frames[0], compareFrames[0], edgeParams))


def printBlobMetrics():

    blobParams = {}
    blobParams['hsv_min'] = (0,110,79) 
    blobParams['hsv_max'] = (24,255,255)
    blobParams['kernel'] = (7,7)
    ballFrames = getFrames('videos/ball.avi', 1)
    bf = ballFrames[0]
    print(getBlobMetrics(bf,bf, blobParams))

def getMSE(i,i1):
    return np.square(np.subtract(i, i1)).mean()


def getPSNR(origImg,cmpImg):
    MSE = getMSE(origImg, cmpImg)
    return 20 * np.log10(255/ MSE**0.5)



def getMetrics(params, origImg, cmpImg):
    metrics = {}
    metrics['PSNR'] = getPSNR(origImg, cmpImg) 
    metrics['MSE'] = getMSE(origImg, cmpImg)
    metrics['SSIM'] = getSSIM(origImg, cmpImg)
    metrics['MSSSIM'] = getMSSSIM(origImg, cmpImg, params.msssim_weights)
    metrics.update(getEdgeMetrics(origImg, cmpImg, params))
    metrics.update(getBlobMetrics(origImg, cmpImg, params))
    return metrics


frame = getFrames('videos/my_video-2.mkv', 1)[0]
cmpFrame = getFrames('videos/newest_test.mp4', 1)[0]

params = Parameters()

print(getMetrics(params, frame, cmpFrame))
