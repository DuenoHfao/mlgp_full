import torch
import torch.optim
import os
import logging
import argparse
import time
from time import strftime, localtime

from preprocessors.zero_dce.utils import dataloader
from preprocessors.zero_dce.utils import loss
from preprocessors.zero_dce.utils import generate_train_test
import preprocessors.zero_dce.model as model

src_path='./data/img_dataset'
train_path='./data/train_data'
val_path='./data/val_data'
test_path='./data/test_data'
split_ratio=[0.7, 0.2, 0.1]
seed=69420
logger = logging.getLogger(__name__)

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = model.enhance_net_nopool().cuda()
    net.apply(weights_init)
    if config.load_pretrain:
        net.load_state_dict(torch.load(config.pretrain_dir))

    train_dataset = []
    for root, dirs, file_names in os.walk(config.lowlight_images_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            try:
                train_dataset.append(dataloader.lowlight(file_path))

            except RuntimeError as rt_e:
                print(f"Error loading image {file_path}. Skipping.")
                formatted_current_time = strftime("%a, %d %b %Y, %H:%M:%S", localtime(time.time()))
                logger.error(f"{formatted_current_time}: {rt_e}")
                continue
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                               shuffle=True, num_workers=config.num_workers, pin_memory=True)

    L_color, L_spa, L_exp, L_TV = loss.ColourConstancyLoss(), loss.SpacialConstancyLoss(), loss.ExposureLoss(16, 0.6), loss.TotalVariationLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    net.train()
    for epoch in range(config.num_epochs):
        try:
            for i, img in enumerate(train_loader):
                img = img.cuda()
                enhanced_1, enhanced, A = net(img)
                loss_total = (
                    200 * L_TV(A) +
                    torch.mean(L_spa(enhanced, img)) +
                    5 * torch.mean(L_color(enhanced)) +
                    10 * torch.mean(L_exp(enhanced))
                )
                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                optimizer.step()

                if (i + 1) % config.display_iter == 0:
                    print(f"Epoch {epoch}, Iter {i+1}: Loss = {loss_total.item():.4f}")
                if (i + 1) % config.snapshot_iter == 0:
                    torch.save(net.state_dict(), os.path.join(config.snapshots_folder, f"Epoch{epoch}.pth"))
                

        except TypeError:
            print(f"End of folder reached")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging.basicConfig(filename=r'./log_files/zero_dce-train.log', level=logging.ERROR)

    generate_train_test.generate_train_val_test(src_path, train_path, val_path, test_path, split_ratio, seed=seed)
    
    parser.add_argument('--lowlight_images_path', type=str, default=train_path)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="./zero_dce/snapshots")
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_dir', type=str, default="./zero_dce/snapshots/Epoch99.pth")
    config = parser.parse_args()

    os.makedirs(config.snapshots_folder, exist_ok=True)
    train(config)
