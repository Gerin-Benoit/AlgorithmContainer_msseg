from monai.networks.nets import UNet, SegResNet
import sys
sys.path.append('../..')
import torch.nn as nn
from unet import *

def get_model(device, in_shape, name='UNet', constrained=False, in_channels=4, out_channels=4, activation = nn.ELU,  inference=False, ssl=False):
    # model = UNet(
    #     dimensions=3,
    #     in_channels=4,
    #     out_channels=3,
    #     channels=(32, 64, 128, 256, 320),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # )
    if name == 'SegResNet':
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=0.2,
        )

    elif name == 'UNet':
        if constrained:
            if inference:
                model = PytorchUNet3D(
                        in_shape,
                        c=None,
                        cout=None,
                        norm_layer=ActNormLP3D,
                        num_classes = out_channels,
                        n_channels = in_channels,
                        activation=activation,
                        device = device
                        )
            else:
                model = PytorchUNet3D(
                    in_shape,
                    c=1,
                    cout=1,
                    norm_layer=ActNormLP3D,
                    num_classes = out_channels,
                    n_channels = in_channels,
                    device=device,
                    ssl=ssl
                    )

        else:
            model = PytorchUNet3D(
                    in_shape,
                    c=None,
                    cout=None,
                    norm_layer = nn.BatchNorm3d,
                    num_classes = out_channels,
                    n_channels = in_channels,
                    activation= activation,
                    device = device,
                    ssl=ssl
                    )
    return model.to(device)
