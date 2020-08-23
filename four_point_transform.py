import cv2
import numpy as np

def order_points(pts):

    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, points, width=0, height=0):
    #upack points
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    
    if (width == 0 and height == 0):
        #compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        mxW = max(int(widthA), int(widthB))
        
        # compute the height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        mxH = max(int(heightA), int(heightB))
    
        # construct a set of destination points to obtain a top
        # down view of the image
        # we specify the points in the smae order
        dst = np.array([[0,0], [mxW -1, 0], [mxW -1, mxH -1], [0, mxH -1]], dtype="float32")
        # compute homography matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (mxW, mxH))
    else:
        dst = np.array([[0,0], [width -1, 0], [width -1, height -1], [0, height -1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
    
    return (warped, M)

def augment(eindex, frame, corners, image_dim, video_frame, frame_dim, image):
    width, height = image_dim[0], image_dim[1]
    frame_width, frame_height = frame_dim[0], frame_dim[1]
    
    (warp, M) = four_point_transform(frame, np.array([corners[eindex][0][0],
                                                corners[eindex][0][1],
                                                corners[eindex][0][2],
                                            corners[eindex][0][3]]), width, height)
    cv2.imshow('warp', warp)
    
    # compute the inverse
    M_inverse = np.linalg.inv(M)
    back_warp = cv2.warpPerspective(video_frame, M_inverse, (frame_width, frame_height))

    # Load two images
    img2 = back_warp
    # create an ROI representing the top left corner
    rows, cols, channels = img2.shape
    roi = image[0:rows, 0:cols]
    
    # create a mask of the logo
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # inverse mask
    mask_inv = cv2.bitwise_not(mask)
    
    # black-out the area of logo in ROI
    image_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    
    # Put logo in ROI and modify the main image
    dst2 = cv2.add(image_bg,img2_fg)
    image[0:rows, 0:cols ] = dst2
    

