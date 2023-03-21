import torch
from torch import nn
import argparse
import wandb
from voxel_uncertainty import voxel_uncertainty
# from datasets.brats.data import get_dataset as get_brats
# from datasets.msseg.data import get_dataset as get_msseg
# from datasets.lung.data import get_dataset as get_lung
# from datasets.chestxray.data import get_dataset as get_chestxray
from scipy.stats import wasserstein_distance
from monai.transforms import Compose, AsDiscrete

from density_unet import DensityUnet
from ensemble import EnsembleUnet

from unet import ActNormLP3D, ActNormLP2D, PytorchUNet3D, PytorchUNet2D



def main(args):
    def get_default_device():
        """ Set device """
        if torch.cuda.is_available():
            print("Got CUDA!")
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    roi_size = (96, 96, 96)
    device = get_default_device()

    # Load dataset
    if args.dataset == 'brats':
        from datasets.brats.data import get_dataset as get_brats
        from datasets.brats.retention_curves import compute_retention_curve
        in_channels, out_channels = 4, 4
        task = '3D'
        in_shape = (in_channels,) + roi_size
        _, _, test_loader = get_brats(batch_size=args.batch_size,
                                      roi_size=roi_size,
                                      root_dir=args.root_dir,
                                      num_workers=args.num_workers,
                                      cache_rate=args.cache_rate,
                                      seed=args.seed,
                                      val_only=True)

    elif args.dataset == 'msseg':
        from datasets.msseg.data import get_dataset as get_msseg
        in_channels, out_channels = 1, 1
        task = '3D'
        in_shape = (in_channels,) + roi_size
        _, _, test_loader = get_msseg(roi_size, args.num_workers)


    elif args.dataset == 'lung':
        from datasets.lung.data import get_dataset as get_lung
        in_channels, out_channels = 1, 1
        task = '3D'
        in_shape = (in_channels,) + roi_size
        _, _, test_loader = get_lung(roi_size, args.num_workers)



    elif args.dataset == 'chestxray':
        from datasets.chestxray.data import get_dataset as get_chestxray
        task = '2D'
        _, _, test_loader = get_chestxray(args.num_workers)

        # Load model
    if task == '3D':
        """
        simple_unet = PytorchUnet3D(in_shape,
                             c=None,
                             norm_layer= nn.BatchNorm3d,
                             num_classes = out_channels,
                             n_channels = in_channels,
                             device
                             )
        """
        if args.density and args.constrained:
            unet = PytorchUNet3D(in_shape,
                                c=None,
                                norm_layer=ActNormLP3D,
                                num_classes=out_channels,
                                n_channels=in_channels,
                                device=device,
                                cout=None).to(device)
            x = torch.rand((1,4,) + roi_size).to(device)
            y = unet(x)
        elif args.density:
            unet = PytorchUNet3D(in_shape,
                                c=None,
                                norm_layer=nn.BatchNorm3d,
                                num_classes=out_channels,
                                n_channels=in_channels,
                                device=device,
                                cout=None).to(device)
        elif args.ensemble:
            unets = [PytorchUNet3D(in_shape, 
                                   c=None, 
                                   norm_layer = nn.BatchNorm3d,
                                   num_classes = out_channels,
                                   n_channels = in_channels,
                                   device = device,
                                   cout=None).to(device) 
                     for i in range(len(args.path_ensemble))]
        
    elif task == '2D':
        """
        simple_unet = PytorchUnet2D(in_shape,
                             c=None, 
                             norm_layer= nn.BatchNorm2d,
                             num_classes = out_channels,
                             n_channels = in_channels,
                             device=device
                             )
        """
        if args.constrained:
            unet = PytorchUNet2D(in_shape,
                                c=None,
                                norm_layer=ActNormLP2D,
                                num_classes=out_channels,
                                n_channels=in_channels,
                                device=device,
                                cout=None
                                ).to(device)
        else:
            unet = PytorchUNet2D(in_shape,
                                c=None,
                                norm_layer=nn.BatchNorm2d,
                                num_classes=out_channels,
                                n_channels=in_channels,
                                device=device,
                                cout=None
                                ).to(device)
        '''
        unets = [PytorchUnet2D(in_shape, 
                               c=None, 
                               norm_layer = nn.BatchNorm2d,
                               num_classes = out_channels,
                               n_channels = in_channels) 
                 for i in range(len(args.path_ensemble))]
        '''
    

    
    #print('keys:', keys)
    #print('keys_scratch',keys_scratch)
    # TO DO : path gmms -> passer d'un chemin vers un dossier à une liste avec 5 chemins (et chemin = None si osef)
    if args.density:
        super_model = DensityUnet(path_gmms=args.path_gmms, path_density_unet=args.path_density_unet, unet=unet, combination='last', K=4).to(device)
        super_model.Unet_init = True
        super_model.eval()
    elif args.ensemble:
        super_model = EnsembleUnet(Unets=unets, paths=args.path_ensemble, uncertainty='variance', reduction='majority_vote', act=nn.Softmax(dim=1))
        super_model.Unets_init = True
        super_model.eval()

    post_trans = Compose(
        [AsDiscrete(argmax=True, to_onehot=out_channels)]
    )
    
    # Test loop
    # Measure uncertainty : 1-Wassertein distance
    
    compute_retention_curve(super_model,
                            roi_size,
                            test_loader,
                            post_trans,
                            args.save_path,
                            args.ensemble,
                            device)

    """
    uncertainty_map1 = ...
    true_error1 = ...
    uncertainty_map2 = ...
    true_error2 = ...

    wass_distance1 = wasserstein_distance(uncertainty_map1,
                                          true_error1)  # check how to use wassertein_distance of scipy
    wass_distance1 = wasserstein_distance(uncertainty_map2, true_error2)
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--wandb_project', type=str, default='uncertainty_measures', help='wandb project name')
    parser.add_argument('--dataset', default="brats", help='dataset to test')
    parser.add_argument('--root_dir', type=str, help='path to data dir')
    parser.add_argument('--seed', type=int, default=42, help='Specify the global random seed')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloaders.')
    parser.add_argument('--cache_rate', type=float, default=1.0, help='cached dataset in RAM')
    parser.add_argument('--batch_size', type=int, default=1, help='size of batch')
    parser.add_argument('--constrained', default=False, action='store_true')
    parser.add_argument('--path_density_unet', type=str, help='path to the weight of the density unet')
    parser.add_argument('--path_gmms', nargs='+', type=str, default=['_', '_', '_', '_', '_'])
    parser.add_argument('--path_ensemble', nargs='+', type=str,
                        help='list of path to the models without constraints to build the ensemble')
    parser.add_argument('--save_path',  type=str,
                        help='path to save the retention_curve')
    parser.add_argument('--ensemble', action='store_true', default=False)
    parser.add_argument('--density', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
