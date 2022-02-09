import cv2
import os
import numpy as np
import argparse

def kp_fast(S,T,N = 50):
    pass

def kp_dog(S,T,N = 50):
    pass

def des_sift(S,T,N = 50):
    pass

def des_brief(S,T,N = 50):
    pass

def JoinSourceTarget(S,T):
    final = np.concatenate((S,T),axis=1)
    return final

parser = argparse.ArgumentParser()

parser.add_argument("-kp","--kp",help= '''Should be either of "fast" or "dog" ''',required=True)
parser.add_argument("-des","--des",help= '''Should be either of "sift" or "brief" ''',required=True)
parser.add_argument("-trans","--trans",help= '''Should be one of "view","scale","rot" or "light" ''',required=True)
parser.add_argument("-nm","--nm",help= '''Number of points to be matched ''',default=50,type=int)

args = parser.parse_args()

source_img_name = args.trans + "_S.ppm"
target_img_name = args.trans + "_T.ppm"
N = args.nm

S = cv2.imread(os.path.join(r"..\data",source_img_name))
T = cv2.imread(os.path.join(r"..\data",target_img_name))

eval()

final = JoinSourceTarget(S,T)

cv2.imshow("Output",final)
k = cv2.waitKey(0)

while True:
    if k == ord('q'):
        break
cv2.destroyAllWindows()

sift = cv2.xfeatures2d.SIFT_create(N)
orb = cv2.ORB_create(N)
#Sift descriptor uses DoG detector
#ORB uses Fast detector
train_keypoints_dog = sift.detect(S,None) #Image,Mask
test_keypoints_dog = sift.detect(T, None)

train_keypoints_fast = orb.detect(S,None)
test_keypoints_fast = orb.detect(T,None)

cv2.drawKeypoints(S, train_keypoints_dog, keypoints_without_size, color = (0, 255, 0))

cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
