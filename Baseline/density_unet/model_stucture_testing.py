import argparse
import os
import monai
import torch
from torch import nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import UNet
import numpy as np
import random
from metrics import dice_metric
from data_load import get_train_dataloader, get_val_dataloader
import wandb
from monai.networks.layers.factories import Act
from divers import *
from unet import *
import pickle
#from torchsummary import summary


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')




num_res_units = 1

seg_model = PytorchUNet()
seg_model.load_state_dict(torch.load('/home/gerinb/Grece/to_test/model.pth'))

device = get_default_device()
#device = 'cpu'

dens_bg = '/home/gerinb/Grece/to_test/GMM_BG_2_components.sav'
dens_fg = '/home/gerinb/Grece/to_test/GMM_FG_2_components.sav'

super_model = TorchFastSuperModel(segmentation_model=seg_model,
                             density_model_BG=dens_bg,
                             density_model_FG=dens_fg, device=device).to(device)
x = torch.zeros((1, 1, 96, 96, 96)).to(device)  # B C X Y Z

y,p = super_model(x)
print(y.shape,p.shape)
exit()
