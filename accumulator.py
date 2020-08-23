import numpy as np
import cv2
from cv2 import aruco
from four_point_transform import four_point_transform
from four_point_transform import augment

flatten = lambda l: [item for sublist in l for item in sublist]

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

width, height = 200, 200
rect = np.zeros((4, 2), dtype="float32")

cap = cv2.VideoCapture(-1)
cap2 = cv2.VideoCapture("_data/blue22.jpg")

while (cap.isOpened()):
    ret, frame = cap.read()
    frame_height, frame_width, frame_channels = frame.shape
    
    ret2, frame2 = cap2.read()
    
    try:
        frame2_height, frame2_width, frame2_channels = frame2.shape
        video_frame = cv2.resize(frame2, (200, 200))
    except AttributeError:
        pass
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,\
                                                           parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    img1 = frame
    
    if (len(corners) > 0):
        ids2 = ids.flatten()
        
        eindex = -1
        vals = [i for i in range(1, 5)]
        found = {1:[False], 2:[False], 3:[False], 4:[False]}
        #rint(type(found[1][0]))
        
        
        for val in ids:
            eindex += 1
            
            for i in vals:
                if (i == val):
                    print("val is ", val)
                    found[val[0]][0] = True
                    found[val[0]].append(eindex)
        
        if (found[1][0]):
            augment(found[1][1], frame, corners, (width, height), video_frame,\
                    (frame_width, frame_height), img1)
            
    cv2.imshow('augmented', img1)
    cv2.imshow('frame_markers', frame_markers)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
                

