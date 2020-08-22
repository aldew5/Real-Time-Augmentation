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
    
