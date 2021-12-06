from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

imageA = cv2.imread("/home/yakorde/capstone/CapstoneSecureComputing/test.jpg")
imageB = cv2.imread("/home/yakorde/capstone/CapstoneSecureComputing/test.jpg")

s = ssim(imageA, imageB)

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

 # Check for same size and ratio and report accordingly
ho, wo, _ = image1.shape
hc, wc, _ = image2.shape
ratio_orig = ho/wo
ratio_comp = hc/wc
dim = (wc, hc)

print(s)