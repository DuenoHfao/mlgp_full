import numpy as np
import math
import cv2

def get_brightness(img_file):
    img_file = img_file.copy().astype(np.float32)
    b, g, r = cv2.split(img_file)
    b = np.mean(b)
    g = np.mean(g)
    r = np.mean(r)
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

def msr(img, WEIGHTS, GaussianSize=3, gamma=0):
    '''
    https://www-spiedigitallibrary-org.remotexs.ntu.edu.sg/conference-proceedings-of-spie/11069/110692N/An-improved-image-enhancement-method-based-on-lab-color-space/10.1117/12.2524449.full
    '''
    img = img.astype(np.float32)

    img_brightness = get_brightness(img)
    for weight_key in WEIGHTS.keys():
        if img_brightness < weight_key:
            weights = WEIGHTS[weight_key]
            break
    
    else:
        return img.astype(np.uint8)
    
    blurred_img = cv2.GaussianBlur(img, (GaussianSize, GaussianSize), 0)
    illumination = np.log1p(img / (blurred_img +1e-7))
    reflection = np.log1p(img) - illumination

    outline_highlights = cv2.addWeighted(illumination, weights[0], reflection, weights[1], gamma)
    add_original_img = cv2.addWeighted(img, 0.5, outline_highlights, 0.5, gamma)
    outline_highlights = cv2.normalize(add_original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return outline_highlights

def colour_correction(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.mean(lab_img[:, :, 1])
    avg_b = np.mean(lab_img[:, :, 2])
    lab_img[:, :, 1] = lab_img[:, :, 1] - avg_a
    lab_img[:, :, 2] = lab_img[:, :, 2] - avg_b

    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    rgb_normalized_img = cv2.normalize(rgb_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return rgb_normalized_img

def process_img(img_path):
    img = cv2.imread(img_path)
    input_weights = {
        70: (0.3, 0.3)
    }
    cv2.imshow("Original Image", img)

    img = msr(img, input_weights)
    img = colour_correction(img)
    return img

if __name__ == "__main__":
    img_path = r"C:\Users\DuenoHfao\Desktop\Work\SCVU\mlgp_full\data\msrcc_test_data\images\Bicycle\2015_00644.jpg"
    

    output_img = process_img(img_path)
    cv2.imshow("Enhanced Image", output_img)
    cv2.waitKey(0)