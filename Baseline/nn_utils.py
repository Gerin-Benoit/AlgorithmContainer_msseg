import os
import torch
from torch import nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from unet import PytorchUNet
import numpy as np
import random

# import for Smooth Layer
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

def replace_conv(module, c):
    '''
    Recursively put SmoothLayer around conv layers in nn.module module.
    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and put SmoothLayers around conv layers
    if type(module)==SmoothLayer:
        return
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.ConvTranspose3d:
            smooth_conv = SmoothLayer(target_attr, c=c)
            setattr(module, attr_str, smooth_conv)
            

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_conv(immediate_child_module, c)
        
def get_model(model_name, device, **kwargs):
    if model_name=='UNet_monai':
        if kwargs['n_blocks']==5:
            channels = (32, 64, 128, 256, 512)
            strides = (2,2,2,2)
        if kwargs['norm']=='group':
            norm=("group", {"num_groups": 4})
        elif kwargs['norm']=='instance':
            norm = 'INSTANCE'
        activation = kwargs['activation']
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=kwargs['kernel_size'],
            up_kernel_size=kwargs['kernel_size'],
            bias=kwargs['bias'],
            channels=channels,
            strides=strides,
            num_res_units=kwargs['n_residuals'],
            act=activation,
            norm=norm).to(device)
        if kwargs['spectral_normalization']:
            replace_conv(model, c=kwargs['c'])
    elif model_name=='UNet':
        if kwargs['norm_layer']=='instance':
            norm_layer = nn.InstanceNorm3d
        elif kwargs['norm_layer']=='batch':
            norm_layer = nn.BatchNorm3d
            
        if kwargs['activation'] == 'ReLU':
            activation = nn.ReLU
        elif kwargs['activation'] == 'ELU':
            activation = nn.ELU
        elif kwargs['activation'] == 'LReLU':
            activation = nn.LeakyReLU
        model = PytorchUNet(
                     c=kwargs['c'],
                     n_channels=1,  # grayscale
                     norm_layer=norm_layer,
                     activation=activation,
                     act_ssl = None,
                     ).to(device)
    """
    elif model_name=='UNETR':
        model = UNETR(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            img_size=roi_size,
            hidden_size=384,
            num_heads=8,
            feature_size=16).to(device)
    """
        
    return model
        

def get_optimizer(optimizer_name, model, **kwargs):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = kwargs['learning_rate'], weight_decay=kwargs['weight_decay'])
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = kwargs['learning_rate'], weight_decay=kwargs['weight_decay'])
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = kwargs['learning_rate'], weight_decay=kwargs['weight_decay'])
        
    return optimizer

def get_scheduler(scheduler_name, optimizer, warmup_steps, **kwargs):
    if scheduler_name=='ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                'max', 
                                                                verbose=False, 
                                                                patience=kwargs['patience'],
                                                                factor=kwargs['factor'])
    elif scheduler_name=='CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs['T_max'], eta_min=kwargs['lr_min'])
    elif scheduler_name=='CosineLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs['max_epochs'], eta_min=kwargs['lr_min'])
    elif scheduler_name=='ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=kwargs['gamma'])
    # Warmup to implement !!
    """
    if warmup_steps>0:
        scheduler = create_lr_scheduler_with_warmup(scheduler,
                                                    warmup_start_value=kwargs['lr_init'],
                                                    warmup_end_value=kwargs['lr_init'] * kwargs['warmup_lr_mult'],
                                                    warmup_duration=warmup_steps)
    """
    
        
    return scheduler



def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_loss_function(args):
    if args.loss == "dice":
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, sigmoid=False, include_background=False)
    elif args.loss == "diceAU":
        loss_function = DiceLossAU(to_onehot_y=True, softmax=True, sigmoid=False, include_background=False,
                                   include_uncertainty_in_denominator=True)
    elif args.loss == "diceAU_no_denominator":
        loss_function = DiceLossAU(to_onehot_y=True, softmax=True, sigmoid=False, include_background=False,
                                   include_uncertainty_in_denominator=False)
       
        
        
"""
Source : https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py
"""

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SmoothLayer(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, c=1):
        super(SmoothLayer, self).__init__()
        self.conv = module
        self.name = name
        self.power_iterations = power_iterations
        self.c = c
        if not self._made_params():
            self._make_params()

    def _update(self):
        u = getattr(self.conv, self.name + "_u")
        v = getattr(self.conv, self.name + "_v")
        w = getattr(self.conv, self.name + "_bar")
        

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.conv, self.name, (w / sigma.expand_as(w)) * self.c)
       

    def _made_params(self):
        try:
            u = getattr(self.conv, self.name + "_u")
            v = getattr(self.conv, self.name + "_v")
            w = getattr(self.conv, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        
        w = getattr(self.conv, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.conv._parameters[self.name]

        self.conv.register_parameter(self.name + "_u", u)
        self.conv.register_parameter(self.name + "_v", v)
        self.conv.register_parameter(self.name + "_bar", w_bar)



    def forward(self, *args):
        self._update()
        return self.conv.forward(*args)
