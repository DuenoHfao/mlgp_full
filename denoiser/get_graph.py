import os
import tensorflow as tf

MODEL_PATH = r"./denoiser/maxim_s3_denoising_sidd"

model = tf.saved_model.load(MODEL_PATH)

print(model.signatures)