import torch
import torchvision
import os

import numpy as np
from PIL import Image
import model

from lowlight_train import test_path

def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    BASE_RESULT_DIR = './zero_dce/result'

    img = Image.open(image_path)
    img = torch.from_numpy(np.asarray(img) / 255.0).float().permute(2, 0, 1).cuda().unsqueeze(0)
    
    net = model.enhance_net_nopool().cuda()
    net.load_state_dict(torch.load('./zero_dce/snapshots/Epoch99.pth'))
    net.eval()
    print(img.shape)
    with torch.no_grad():
        try:
            _, enhanced, _ = net(img)
        except RuntimeError as e:
            print(image_path)
            
            print(e)
            exit()
    
    dir_name = os.path.dirname(image_path.replace('\\', '/')).split('/')[-1]
    result_folder_path = os.path.join(BASE_RESULT_DIR, dir_name)
    result_path = os.path.join(result_folder_path, os.path.basename(image_path))
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torchvision.utils.save_image(enhanced, result_path)
    print(f"Saved enhanced image to {result_path}")

if __name__ == '__main__':
    test_dir = test_path
    os.makedirs(test_dir, exist_ok=True)

    for (root, dirs, files) in os.walk(test_dir):
        if files == []:
            continue
        for file_name in files:
            img_path = os.path.join(root, file_name)
            lowlight(img_path)

