import math
import cv2


def distance(p1, p2, p3, p4, p5, p6, p7, p8): 
    centroid1 = ((p1[0] + p2[0] + p3[0] + p4[0])/4,\
                (p1[1] + p2[1] + p3[1] + p4[1])/4)\
    
    centroid2 = ((p5[0] + p6[0] + p7[0] + p8[0])/4,\
                (p5[1] + p6[1] + p7[1] + p8[1])/4)
    
    dist = math.sqrt(abs(centroid1[0] - centroid2[0]) ** 2 + abs(centroid1[1] - centroid2[1])\
                     ** 2)
    return dist




            
            
