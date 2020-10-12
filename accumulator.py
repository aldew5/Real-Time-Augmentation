import numpy as np
import cv2
from cv2 import aruco
from four_point_transform import four_point_transform
from four_point_transform import augment
from distance import distance


flatten = lambda l: [item for sublist in l for item in sublist]

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

width, height = 200, 200
rect = np.zeros((4, 2), dtype="float32")

cap = cv2.VideoCapture(-1)

frame2 = cv2.imread("_data/blue22.jpg", cv2.IMREAD_COLOR)
frame3 = cv2.imread("_data/green.jpeg", cv2.IMREAD_COLOR)
frame4 = cv2.imread("_data/green-blue.jpeg", cv2.IMREAD_COLOR)
frame5 = cv2.imread("_data/white.png", cv2.IMREAD_COLOR)
ashbury = cv2.imread("_data/ashbury.jpeg", cv2.IMREAD_COLOR)

video_frame = cv2.resize(frame2, (200, 200))
video_frame2 = cv2.resize(frame3, (200, 200))
video_frame3 = cv2.resize(frame4, (200, 200))
video_frame4 = cv2.resize(frame5, (200, 200))
ashbury_video = cv2.resize(ashbury, (200, 200))

color = ""
display = {1: True, 2: True}

while (cap.isOpened()):
    ret, frame = cap.read()
    frame_height, frame_width, frame_channels = frame.shape
    
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
                    #print("val is ", val)
                    found[val[0]][0] = True
                    found[val[0]].append(eindex)
                    
        
                    
        if (found[1][0]):
            # also found the accumulator
            if (found[3][0]):
                tl = corners[found[1][1]][0][0]
                tr = corners[found[1][1]][0][1]
                br = corners[found[1][1]][0][2]
                bl = corners[found[1][1]][0][3]
                
                tl2 = corners[found[3][1]][0][0]
                tr2 = corners[found[3][1]][0][1]
                br2 = corners[found[3][1]][0][2]
                bl2 = corners[found[3][1]][0][3]
                
                d = distance(tl, tr, br, bl, tl2, tr2, br2, bl2)
                
                # the distance is small enough
                if (d <= 200):
                    # accumualtor is white
                    if (color == ""):
                        color = "blue"
                    # accumulate the colors
                    elif (color == "green"):
                        color = "blue-green"
                    display[1] = False
                    
                elif (display[1]):
                    augment(found[1][1], frame, corners, (width, height), video_frame,\
                        (frame_width, frame_height), img1)
            # didn't find the accumulator (augment normally)
            elif (display[1]):
                augment(found[1][1], frame, corners, (width, height), video_frame,\
                        (frame_width, frame_height), img1)
                
                
                
        if (found[2][0]):
            if (found[3][0]):
                tl = corners[found[2][1]][0][0]
                tr = corners[found[2][1]][0][1]
                br = corners[found[2][1]][0][2]
                bl = corners[found[2][1]][0][3]
                
                tl2 = corners[found[3][1]][0][0]
                tr2 = corners[found[3][1]][0][1]
                br2 = corners[found[3][1]][0][2]
                bl2 = corners[found[3][1]][0][3]
                
                d = distance(tl, tr, br, bl, tl2, tr2, br2, bl2)
                
                if (d <= 200):
                    if (color == ""):
                        color = "green"
                    elif (color == "blue"):
                        color = "blue-green"
                        
                    display[2] = False
                elif (display[2]):
                    augment(found[2][1], frame, corners, (width, height), video_frame2,\
                        (frame_width, frame_height), img1)
            
            elif (display[2]):
                augment(found[2][1], frame, corners, (width, height), video_frame2,\
                        (frame_width, frame_height), img1)
        
        if (not found[3][0]):
            display[1] = True
            display[2] = True
            color = ""
        else:
            if (color == ""):
                augment(found[3][1], frame, corners, (width, height), video_frame4,\
                            (frame_width, frame_height), img1)
            elif (color == "blue"):
                augment(found[3][1], frame, corners, (width, height), video_frame,\
                            (frame_width, frame_height), img1)
            elif (color == "green"):
                augment(found[3][1], frame, corners, (width, height), video_frame2,\
                        (frame_width, frame_height), img1)
            else:
                augment(found[3][1], frame, corners, (width, height), video_frame3,\
                        (frame_width, frame_height), img1)
        if (found[4][0]):
            tl = corners[found[4][1]][0][0]
            tr = corners[found[4][1]][0][1]
            br = corners[found[4][1]][0][2]
            bl = corners[found[4][1]][0][3]
        
            augment(found[4][1], frame, corners, (width, height), ashbury_video,\
                            (frame_width, frame_height), img1)
        
      
            
    cv2.imshow('augmented', img1)
    cv2.imshow('frame_markers', frame_markers)
    #print("color is ", color)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
                
