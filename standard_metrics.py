from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2



def getSSIM(origImg, cmpImg):
    return  ssim(origImg, cmpImg,data_range=cmpImg.max() - cmpImg.min(), multichannel=True)


def getMSSSIM(origImg, cmpImg, params):
    weights = params[0]
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

def getMSE(i,i1):
    return np.square(np.subtract(i, i1)).mean()


def getPSNR(origImg,cmpImg):
    MSE = getMSE(origImg, cmpImg)
    return 20 * np.log10(255/ MSE**0.5)


