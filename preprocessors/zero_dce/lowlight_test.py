import torch
import torchvision
import os
import time
from time import localtime, strftime
import logging

import numpy as np
from PIL import Image
from preprocessors.zero_dce.model import enhance_net_nopool
from preprocessors.zero_dce.lowlight_train import test_path



def lowlight(image_path, save_file=True, save_path=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    img = Image.open(image_path).convert('RGB')
    try:
        img = torch.from_numpy(np.asarray(img) / 255.0).float().permute(2, 0, 1).cuda().unsqueeze(0)
    
        net = enhance_net_nopool().cuda()
        net.load_state_dict(torch.load('./preprocessors/zero_dce/snapshots/Epoch99.pth'))
        net.eval()
        with torch.no_grad():
            _, enhanced, _ = net(img)
    
    except RuntimeError as e:
        formatted_time = strftime("%a, %d %b %Y %H:%M:%S", localtime(time.time()))
        logger.error(f"{formatted_time}: Error processing image {image_path}: {e}")
        print(f"Error processing image {image_path}")
        return
    
    if not save_file:
        return enhanced
    
    if save_path == None:
        raise ValueError("save_path is empty. Please provide a valid path.")
    
    dir_name = os.path.dirname(image_path.replace('\\', '/')).split('/')[-1]
    result_folder_path = os.path.join(save_path, dir_name)
    result_path = os.path.join(result_folder_path, os.path.basename(image_path))
    os.makedirs(result_folder_path, exist_ok=True)
    torchvision.utils.save_image(enhanced, result_path)
    print(f"{result_path}") # Saved enhanced image to {result_path}


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./log_files/zero_dce-test.log', level=logging.ERROR)
    test_dir = test_path
    os.makedirs(test_dir, exist_ok=True)

    for (root, dirs, files) in os.walk(test_dir):
        if files == []:
            continue
        for file_name in files:
            img_path = os.path.join(root, file_name)
            lowlight(img_path)

