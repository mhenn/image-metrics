import cv2
import numpy as np

def foo1(img):
    return img.sum() / 255

def foo2(img):
    a = np.where(img > 0, 1, 0)
    return a.sum()

def foo3(img):
    c = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] > 0:
                c += 1
    return c

if __name__ == "__main__":
    img = np.random.random(size=(640, 480)) * 255
    img = img.astype(np.uint8)
    foo2(img)
    #print(img)
