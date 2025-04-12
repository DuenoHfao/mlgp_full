import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

class L_color(nn.Module):
    def forward(self, x):
        mr, mg, mb = torch.split(torch.mean(x, [2, 3], keepdim=True), 1, dim=1)
        return torch.sqrt((mr - mg)**2 + (mr - mb)**2 + (mb - mg)**2)

class L_spa(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(4)
        def kernel(k): return torch.FloatTensor(k).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(kernel([[0,0,0],[-1,1,0],[0,0,0]]), requires_grad=False)
        self.weight_right = nn.Parameter(kernel([[0,0,0],[0,1,-1],[0,0,0]]), requires_grad=False)
        self.weight_up = nn.Parameter(kernel([[0,-1,0],[0,1,0],[0,0,0]]), requires_grad=False)
        self.weight_down = nn.Parameter(kernel([[0,0,0],[0,1,0],[0,-1,0]]), requires_grad=False)

    def forward(self, org, enhance):
        org_pool = self.pool(torch.mean(org, 1, keepdim=True))
        enh_pool = self.pool(torch.mean(enhance, 1, keepdim=True))
        w_diff = torch.max(torch.tensor([0.5]).cuda(), 1 + 10000 * torch.min(org_pool - 0.3, torch.zeros_like(org_pool)))
        E = sum((F.conv2d(org_pool, w, padding=1) - F.conv2d(enh_pool, w, padding=1))**2
                for w in [self.weight_left, self.weight_right, self.weight_up, self.weight_down])
        return E

class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        mean = self.pool(torch.mean(x, 1, keepdim=True))
        return torch.mean((mean - self.mean_val)**2)

class L_TV(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        h_tv = torch.sum((x[:, :, 1:, :] - x[:, :, :-1, :])**2)
        w_tv = torch.sum((x[:, :, :, 1:] - x[:, :, :, :-1])**2)
        return self.weight * 2 * (h_tv / (x.size(2)-1) / x.size(3) + w_tv / x.size(2) / (x.size(3)-1)) / x.size(0)

class Sa_Loss(nn.Module):
    def forward(self, x):
        r, g, b = torch.split(x, 1, dim=1)
        mr, mg, mb = torch.split(torch.mean(x, [2, 3], keepdim=True), 1, dim=1)
        return torch.mean(torch.sqrt((r - mr)**2 + (g - mg)**2 + (b - mb)**2))

class perception_loss(nn.Module):
    def __init__(self):
        super().__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_4_3_
