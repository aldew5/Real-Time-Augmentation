import numpy as np
import cv2
from cv2 import aruco

flatten = lambda l: [item for sublist in l for item in sublist]

speed_sign_in = cv2.imread('_data/speed_80.bmp',cv2.IMREAD_COLOR)
speed_sign = cv2.resize(speed_sign_in, (200, 200))

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

width, height = 200, 200
dst = np.array([[0,0],[width - 1, 0],[width - 1, height - 1],[0, height - 1],], dtype='float32')
rect = np.zeros((4, 2), dtype = "float32")

cap = cv2.VideoCapture(-1)
cap2 =  cv2.VideoCapture('_data/vtest.avi')

while(cap.isOpened() and cap2.isOpened()):
    ret, frame = cap.read()
    frame_height, frame_width, frame_channels = frame.shape
    ret2, frame2 = cap2.read()
    
    try: 
        frame2_height, frame2_width, frame2_channels = frame2.shape
        video_frame = cv2.resize(frame2, (200, 200))
    except AttributeError:
        pass

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #print(len(corners))
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    img1 = frame


    if (len(corners) > 0):
        #print(type(ids))
        #print(ids)
        #print(len(ids))
        ids2 = ids.flatten()
        #print(ids2)
        #print(len(ids2))
        found_val = False
        eindex = -1
        for val in ids2:
            eindex = eindex + 1
            if (val == 2):
                found_val = True
                break
        #print(found_val, eindex)

        if(found_val):
            # find four corners of the first pattern
            rect[0] = corners[eindex][0][0]
            rect[1] = corners[eindex][0][1]
            rect[2] = corners[eindex][0][2]
            rect[3] = corners[eindex][0][3]

            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(frame, M, (width, height))
            cv2.imshow('warp', warp)
        
            M_inverse = np.linalg.inv(M)
            #back_warp = cv2.warpPerspective(speed_sign, M_inverse, (frame_width, frame_height))
            back_warp = cv2.warpPerspective(video_frame, M_inverse, (frame_width, frame_height))
        
            # Load two images
            img2 = back_warp
            # I want to put logo on top-left corner, So I create a ROI
            rows,cols,channels = img2.shape
            roi = img1[0:rows, 0:cols]
            # Now create a mask of logo and create its inverse mask also
            img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
            # Put logo in ROI and modify the main image
            dst2 = cv2.add(img1_bg,img2_fg)
            img1[0:rows, 0:cols ] = dst2

    cv2.imshow('augmented', img1)
    cv2.imshow('frame_markers', frame_markers)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

