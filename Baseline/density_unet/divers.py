import warnings
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.distributions as D
import pickle
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks import ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export
from unet import *


class TorchFastSuperModel(nn.Module):

    def __init__(
            self,
            segmentation_model: PytorchUNet,
            density_model_BG: str,
            density_model_FG: str,
            device='cuda'
    ) -> None:
        super().__init__()
        self.device = device
        ''' Segmentation '''
        self.segmentation_model = segmentation_model
        ''' GMM '''
        self.log_prob_scaling_factor = 200
        if density_model_BG is not None and density_model_FG is not None:
            GMM_BG = pickle.load(open(density_model_BG, 'rb'))  # density model should not be on GPU !
            self.weights_BG = (torch.from_numpy(GMM_BG.weights_) / 2).to(device)
            self.K_BG = torch.numel(self.weights_BG)
            self.means_BG = torch.from_numpy(GMM_BG.means_).to(device)
            self.cholesky_BG = [torch.from_numpy(np.linalg.cholesky(GMM_BG.covariances_)[i]).to(device) for i in
                                range(self.K_BG)]
            self.components_BG = []
            GMM_FG = pickle.load(open(density_model_FG, 'rb'))  # density model should not be on GPU !
            self.weights_FG = (torch.from_numpy(GMM_FG.weights_) / 2).to(device)
            self.K_FG = torch.numel(self.weights_FG)
            self.means_FG = torch.from_numpy(GMM_FG.means_).to(device)
            self.cholesky_FG = [torch.from_numpy(np.linalg.cholesky(GMM_FG.covariances_)[i]).to(device) for i in
                                range(self.K_FG)]
            self.components_FG = []
            self.n_features = self.means_BG.shape[1]

            for i in range(self.K_BG):
                self.components_BG.append(D.MultivariateNormal(loc=self.means_BG[i, :],
                                                               scale_tril=self.cholesky_BG[i]))
            for i in range(self.K_FG):
                self.components_FG.append(D.MultivariateNormal(loc=self.means_FG[i, :],
                                                               scale_tril=self.cholesky_FG[i]))

    def forward(self, inputs) -> Tuple:
        seg_output, hook = self.segmentation_model.forward_all(inputs)

        feature_map = hook.permute((0, 2, 3, 4, 1))  # channel dim last
        B, X, Y, Z, C = feature_map.shape
        dens_pred = torch.zeros((B, X, Y, Z)).to(self.device)
        for k in range(self.K_BG):
            for frac in range(Z - 1):
                dens_pred[:, :, :, frac:frac + 1] += self.weights_BG[k] * torch.exp(
                    self.components_BG[k].log_prob(feature_map[:, :, :, frac:frac + 1, :])/self.log_prob_scaling_factor)
            # dens_pred[:, :, :, 32:64] += self.weights_BG[k] * self.components_BG[k].log_prob(frac_2)
            # dens_pred[:, :, :, 64:96] += self.weights_BG[k] * self.components_BG[k].log_prob(frac_3)
        for k in range(self.K_FG):
            for frac in range(Z - 1):
                dens_pred[:, :, :, frac:frac + 1] += self.weights_FG[k] * torch.exp(
                    self.components_FG[k].log_prob(feature_map[:, :, :, frac:frac + 1, :])/self.log_prob_scaling_factor)
            # dens_pred += self.weights_FG[k] * self.components_FG[k].log_prob(feature_map)
            # dens_pred[:, :, :, 32:64] += self.weights_FG[k] * self.components_FG[k].log_prob(frac_2)
            # dens_pred[:, :, :, 64:96] += self.weights_FG[k] * self.components_FG[k].log_prob(frac_3)


        dens_output = torch.unsqueeze(dens_pred, dim=1)

        return seg_output, dens_output

    def get_feature_maps(self, inputs):
        hook = self.segmentation_model.get_features_map(inputs)
        return hook


class FastSuperModel(nn.Module):

    def __init__(
            self,
            segmentation_model,
            num_res_units,
            density_model,
            device,
    ) -> None:
        super().__init__()
        ''' Segmentation '''
        self.segmentation_model = segmentation_model
        self.num_res_units = num_res_units
        ''' GMM '''
        if density_model is not None:
            GMM = pickle.load(open(density_model, 'rb'))  # density model should not be on GPU !
            self.weights = torch.from_numpy(GMM.weights_).to(device)
            self.means = torch.from_numpy(GMM.means_).to(device)
            self.covariances = torch.from_numpy(GMM.covariances_).to(device)
            self.K = torch.numel(self.weights)
            self.n_features = self.means.shape[1]
            self.components = []
            for i in range(self.K):
                self.components.append(D.MultivariateNormal(loc=self.means[i, :],
                                                            covariance_matrix=self.covariances[i, :, :]))

            self.density_upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.HOOKS = {}
        self.count = 0

        def count_number_of_activation(net):
            out = [0]

            def count_recursive(net, out):
                for name, layer in net._modules.items():
                    if isinstance(layer, (nn.Sequential, SkipConnection, Convolution, ResidualUnit)):
                        count_recursive(layer, out)
                    else:  # residual presents
                        if name == 'A':
                            out[0] += 1

            count_recursive(net, out)
            return out[0]

        def get_activation(name='last'):
            def hook_fn(m, i, o):
                self.HOOKS[
                    name] = o.detach()  # .detach()  #.detach() disconnect the item from the current graph. It allows to
                # isolate both model graph.

            return hook_fn

        def get_last_fm(net, n_res):
            def get_recursive(net, count, target):

                for name, layer in net._modules.items():
                    # If it is a sequential, don't register a hook on it
                    # but recursively register hook on all it's module children
                    if isinstance(layer, (nn.Sequential, Convolution,
                                          SkipConnection, ResidualUnit)):
                        get_recursive(layer, count, target)
                    else:
                        # it's a non sequential. Register a hook
                        if name == 'A':
                            if count[0] == target:
                                layer.register_forward_hook(get_activation())
                                count[0] += 1
                            else:
                                count[0] += 1

            target = count_number_of_activation(net)
            if n_res == 0:
                target -= 0
            else:
                target -= 1  # 1
            count = [1]
            get_recursive(net, count, target)

        get_last_fm(self.segmentation_model, self.num_res_units)  # place hook

    def forward(self, inputs) -> Tuple:
        seg_output = self.segmentation_model(inputs)
        selected_hook = self.HOOKS['last']

        feature_map = selected_hook.permute((0, 2, 3, 4, 1))  # channel dim last
        dens_pred = 0
        for k in range(self.K):
            dens_pred += self.weights[k] * self.components[k].log_prob(feature_map)
        dens_pred = torch.unsqueeze(dens_pred, dim=1)
        dens_output = self.density_upsampling(dens_pred)
        return seg_output, dens_output

    def get_feature_maps(self, inputs):
        _ = self.segmentation_model(inputs)
        selected_hook = self.HOOKS['last']
        return selected_hook


class ErrorModeling(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernel_size: Union[Sequence[int], int] = 3,
            act: Union[Tuple, str] = Act.PRELU,
            norm: Union[Tuple, str] = Norm.INSTANCE,
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _up_transformation(in_channel: int, out_channel: int, kernel_size, stride: int, act, norm, dropout, bias,
                               adn_ordering) -> nn.Module:
            mod = Convolution(
                3,
                in_channel,
                out_channel,
                strides=stride,
                kernel_size=kernel_size,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                is_transposed=True,
                adn_ordering=adn_ordering,
            )
            return mod

        self.transformation_1 = _up_transformation(self.in_channels, self.channels[0], self.kernel_size, self.strides,
                                                   self.act, self.norm, self.dropout, self.bias, self.adn_ordering)
        self.transformation_2 = _up_transformation(self.channels[0], self.channels[1], self.kernel_size, self.strides,
                                                   self.act, self.norm, self.dropout, self.bias, self.adn_ordering)
        self.transformation_3 = _up_transformation(self.channels[1], self.out_channels, self.kernel_size, self.strides,
                                                   Act.RELU, self.norm, self.dropout, self.bias, self.adn_ordering)

    def forward(self, x) -> torch.Tensor:
        hook1, hook2, hook3 = x
        fusion2 = self.transformation_1(hook1) + hook2
        fusion3 = self.transformation_2(fusion2) + hook3
        output = self.transformation_3(fusion3)
        return output
