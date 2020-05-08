import cv2
import numpy as np 

def getFrames(path,amount):

    cap = cv2.VideoCapture(path)
    
    frames = []
    count =0

    while True and count < amount:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
        count +=1

    return np.asarray(frames)


