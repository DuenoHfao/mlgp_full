import torch
import torchvision
import os
import sys
import glob
import time
import numpy as np
from PIL import Image
from utils import dataloader
import model


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1).cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))

    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    print("Inference time:", time.time() - start)

    result_path = image_path.replace('test_data', 'result')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
    with torch.no_grad():
        filePath = 'data/test_data/'
        for file_name in os.listdir(filePath):
            test_list = glob.glob(os.path.join(filePath, file_name, '*'))
            for image in test_list:
                print("Processing:", image)
                lowlight(image)