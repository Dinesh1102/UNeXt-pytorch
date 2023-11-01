import argparse
import os
from glob import glob

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import kornia as K
import torchvision
import torch.nn.functional as F
from matplotlib import pyplot as plt

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from archs import UNext

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--dataset', default='isic',
                        help='dataset name')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = vars(parse_args())
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', args['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=img_ids,
        img_dir=os.path.join('inputs', args['dataset'], 'images'),
        mask_dir=os.path.join('inputs', args['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    # for c in range(config['num_classes']):
    os.makedirs(os.path.join('outputs_masks'), exist_ok=True)
    os.makedirs(os.path.join('outputs_boundaries'), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)


            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            # output = torch.sigmoid(output).cpu().numpy()    #gradient flow breaks here
            output = torch.sigmoid(output)

            x_sobel: torch.Tensor = K.filters.sobel(output)
            
            kernel_size = 3  # Adjust the kernel size for the desired width
            x_sobel = F.conv2d(x_sobel, torch.ones(1, 1, kernel_size, kernel_size).to(x_sobel.device), padding=kernel_size//2)
            input_np = x_sobel.cpu().detach().numpy()
            input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
            input_np = input_np.transpose(0,2,3,1)
            
            output = output.squeeze(1).cpu().detach().numpy()
            output = (output - output.min()) / (output.max() - output.min())

            for i in range(len(x_sobel)):
              cv2.imwrite(os.path.join('outputs_masks',meta['img_id'][i] + '.png'),(output[i]*255).astype('uint8'))
              cv2.imwrite(os.path.join('outputs_boundaries',meta['img_id'][i] + '.png'),(input_np[i]*255).astype('uint8'))
            

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
