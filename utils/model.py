import torch
import torch.nn as nn
import numpy as np
import random
import cv2
from typing import Literal, OrderedDict

from load_data import LoadDataset



class Model(nn.Module, LoadDataset):
    def __init__(self):
        nn.Module.__init__(self)
        device = (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
        self.device = device

        FILTER_NUMBER = 32
        self.flatten = nn.Flatten()
        # nn.Conv2d(input_dim, num_output_channels, kernel_size, stride, padding, bias=True)
        self.conv1 = nn.Conv2d(3, FILTER_NUMBER, kernel_size=3, stride=1, padding=1, bias=True)

    def set_cv2_error_handler(self, error_handler: Literal["default", "silent"] = "default"):
        def silent_error_handler(*args, **kwargs):
            pass
        
        if error_handler == "silent":
            cv2.redirectError(silent_error_handler)
        elif error_handler == "default":
            cv2.redirectError(cv2.utils.logging.error)
        else:
            raise ValueError("error_handler must be either 'default' or 'silent'")
        
        self.error_handler = error_handler
        
    def set_seed(self, seed=69420):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    model = Model()
    model = model.to(model.device)
    print(f"Model is running on {model.device}")

    # model.load_dataset()