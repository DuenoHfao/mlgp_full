import cv2
import numpy as np

def apply_msr(image, scales=[15, 80, 250], weight=[1/3, 1/3, 1/3]):
    """
    Apply Multi-Scale Retinex (MSR) to an image.
    """
    def single_scale_retinex(img, sigma):
        gaussian = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex = np.log1p(img) - np.log1p(gaussian + 1e-6)
        return retinex

    img = image.astype(np.float32) + 1.0
    msr_result = np.zeros_like(img)
    for scale, w in zip(scales, weight):
        msr_result += w * single_scale_retinex(img, scale)

    msr_result = cv2.normalize(msr_result, None, 0, 255, cv2.NORM_MINMAX)
    return msr_result.astype(np.uint8)

def apply_colour_correction(image):
    """
    Apply colour correction to an image.
    """
    img_float = image.astype(np.float32)
    sum_channels = np.sum(img_float, axis=2, keepdims=True)
    sum_channels[sum_channels == 0] = 1  # Avoid division by zero
    corrected_img = img_float / sum_channels * 255.0
    corrected_img = np.clip(corrected_img, 0, 255)
    return corrected_img.astype(np.uint8)

def process_image(image_path):
    """
    Process an image by applying MSR and colour correction.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Apply MSR
    msr_image = apply_msr(image)

    # Apply colour correction
    corrected_image = apply_colour_correction(msr_image)

    return corrected_image