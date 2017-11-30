import cv2
import numpy as np

target = cv2.imread('1.png',0)
target = target[0:260, 0:260]

pi_1 = cv2.imread('2.png',0)
pi_1 = pi_1[0:260, 0:260]

pi_2 = cv2.imread('3.png',0)
pi_2 = pi_2[0:260, 0:260]

pi_3 = cv2.imread('4.png',0)
pi_3 = pi_3[0:260, 0:260]

policy_set = [pi_1,pi_2,pi_3]
mse_policy = dict()


def mse(imageA, imageB):
	
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	return err
 
if __name__ == '__main__':
    #print target.shape
    
    for p in policy_set:
        err = mse(target,p)
        mse_policy[err] = p
        
    min_err = min(mse_policy.keys())
    best_match = mse_policy[min_err]

    cv2.imshow('target',target)
    cv2.imshow('best_match',best_match)
    
    cv2.waitKey(1000)    
    