from keras.layers import TFSMLayer
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
import os
import cv2

MODEL_PATH = r'./denoiser/maxim_s3_denoising_sidd'
layer = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

def image_denoising(image_path):
    image = Image.open(image_path).convert("RGB")
    img_size = image.size
    image = np.array(image, dtype=np.float32) / 255.0
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))
    image = tf.expand_dims(image, 0)

    predictions_image_tensor = layer(image)
    predictions_image_list = list(predictions_image_tensor.values())
    for image_tensor in predictions_image_list:
        output_image = tf.squeeze(image_tensor, 0).numpy()
        output_image = (output_image * 255).astype(np.uint8)

        output_image = Image.fromarray(output_image)
        output_image = output_image.resize(img_size, resample=Image.Resampling.BICUBIC)
        opencv_output = np.array(output_image)[:,:,::-1].copy()
        cv2.imshow("denoised_scaled", opencv_output)
        cv2.waitKey(0)



image_denoising(r'./data/test_data/Bicycle/2015_00009.jpg')


# # Step 3: Load and preprocess the image
# url = "https://people.sc.fsu.edu/~jburkardt/c_src/image_denoise/balloons_noisy.png"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# original_size = image.size 

# image = np.array(image, dtype=np.float32) / 255.0  # normalize
# image = tf.convert_to_tensor(image)
# image = tf.image.resize(image, (256, 256))
# image = tf.expand_dims(image, 0)  # add batch dimension
# print(image.shape)

# # Step 4: Run inference
# predictions = layer(image)
# predictions = list(predictions.values())[0]

# # Optional: Convert prediction to image
# output_image = tf.squeeze(predictions, 0).numpy()
# output_image = (output_image * 255).astype(np.uint8)

# output_image = Image.fromarray(output_image)
# output_image = output_image.resize(original_size, resample=Image.Resampling.BICUBIC)
# output_image.save("denoised_output.png")
