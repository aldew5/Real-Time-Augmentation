import numpy as np
import cv2
from cv2 import aruco
from four_point_transform import four_point_transform

flatten = lambda l: [item for sublist in l for item in sublist]

speed_sign_in = cv2.imread('_data/speed_80.bmp',cv2.IMREAD_COLOR)
speed_sign = cv2.resize(speed_sign_in, (200, 200))

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

width, height = 200, 200
#dst = np.array([[0,0],[width - 1, 0],[width - 1, height - 1],[0, height - 1],], dtype='float32')
rect = np.zeros((4, 2), dtype = "float32")

# external camera video capture object
cap = cv2.VideoCapture(-1)
# the augmentation video
cap2 =  cv2.VideoCapture('_data/vtest.avi')

# using both captures
while(cap.isOpened() and cap2.isOpened()):
    # frame object
    ret, frame = cap.read()
    # dimensions
    frame_height, frame_width, frame_channels = frame.shape
    
    # load the park video 
    ret2, frame2 = cap2.read()
    
    try:
        # get the dimensions of the park video
        frame2_height, frame2_width, frame2_channels = frame2.shape
        # resize to fit the marker
        video_frame = cv2.resize(frame2, (200, 200))
    except AttributeError:
        pass
    
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # draw the corners and the ids on the detected markers
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    # save the frame of the real-time video
    img1 = frame
    
    # detecting at least one corner of a marker
    if (len(corners) > 0):
        # convert to a one-dimensional array
        ids2 = ids.flatten()
        
        found_val = False
        eindex = -1
        # look for a marker with the specified id
        for val in ids2:
            eindex += 1
            if (val == 2):
                found_val = True
                break
        
        # found the specified marker
        if(found_val):
            (warp, M) = four_point_transform(frame, np.array([corners[eindex][0][0],
                                                corners[eindex][0][1],
                                                corners[eindex][0][2],
                                                corners[eindex][0][3]]), width, height)
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
    #cv2.imshow('frame_markers', frame_markersarkers)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

