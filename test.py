from __future__ import division
from skimage.measure import compare_psnr, compare_ssim
import time
import cv2

img1 = cv2.imread('demo.jpg')
img2 = cv2.GaussianBlur(img1, (5,5), 1)

N = 100
t = time.time()
for i in range(N):
    compare_psnr(img1, img2)
t_psnr = (time.time() - t)/N

t = time.time()
for i in range(N):
    compare_ssim(img1, img2, multichannel=True)
t_ssim = (time.time() - t)/N

print('psnr time: %f'%t_psnr)
print('ssim time: %f'%t_ssim)

