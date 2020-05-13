from frames import * 
from edgeMetrics import *
from blobMetrics import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

#import skvideo.measure.msssim as msssim
from tensorflow.image import ssim_multiscale as msssim 
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2



def getSSIM(origImg, cmpImg):
    return  ssim(origImg, cmpImg,data_range=cmpImg.max() - cmpImg.min(), multichannel=True)

def getMSSSIM(origImg, cmpImg):
    return msssim(origImg, cmpImg,255).numpy()


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


#printBlobMetrics()
#printEdgeMetrics()


frame = getFrames('videos/my_video-2.mkv', 1)[0]
cmpFrame = getFrames('videos/newest_test.mp4', 1)[0]





#playDetectedFrames('./videos/ball.avi', (0,110,79),(24,255,255))
#print(getPSNR(frame,cmpFrame))
#print(getMSE(frame,cmpFrame))
#print(getSSIM(frame,cmpFrame))
print(getMSSSIM(frame, cmpFrame))
#cv2.imwrite('imgs/ball.jpg', ballFrames[300])
#cv2.waitKey()

