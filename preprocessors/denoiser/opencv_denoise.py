import os
import cv2

import numpy as np

img_path = r'./data/test_data/Bicycle/2015_00009.jpg'

def artificial_noiser(img):

    noise = np.random.normal(10,50, img.shape)

    img_noised = img + noise
    img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)

    stitched_comparison = np.hstack((img, img_noised))

    cv2.imshow("post_noised", stitched_comparison)
    cv2.waitKey(0)


image = cv2.imread(img_path)
artificial_noiser(image)