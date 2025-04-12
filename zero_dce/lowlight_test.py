import torch
import torchvision
import os
import glob
import numpy as np
from PIL import Image
import model
from utils import dataloader

def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    img = Image.open(image_path)
    img = torch.from_numpy(np.asarray(img) / 255.0).float().permute(2, 0, 1).cuda().unsqueeze(0)

    net = model.enhance_net_nopool().cuda()
    net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
    net.eval()

    with torch.no_grad():
        _, enhanced, _ = net(img)

    result_path = image_path.replace('test_data', 'result')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torchvision.utils.save_image(enhanced, result_path)
    print(f"Saved enhanced image to {result_path}")

if __name__ == '__main__':
    test_dir = 'data/test_data/'
    for subdir in os.listdir(test_dir):
        for img_path in glob.glob(os.path.join(test_dir, subdir, '*')):
            print(img_path)
            lowlight(img_path)
