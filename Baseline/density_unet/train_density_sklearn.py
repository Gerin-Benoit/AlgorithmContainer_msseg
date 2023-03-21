import os
from monai.data import DataLoader, decollate_batch
import wandb
from monai.utils import set_determinism
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from density_unet import *
from torchviz import make_dot
import faiss
# import pandas as pd

from datasets.brats.features import prepare_features as prepare_features_brats
from datasets.brats.data import get_dataset as get_dataset_brats
from datasets.brats.model import get_model as get_model_brats
import glob

parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# login
parser.add_argument('--wandb_project', type=str, default='density', help='wandb project name')
parser.add_argument('--name', default="idiot without a name", help='Wandb run name')

# architecture
parser.add_argument('--prepare', default=False, action='store_true')
parser.add_argument('--levels', nargs='+', type=int, default=[1],
                    help='list of the levels to fit a GMM (pass 1 2 3 4 5 for all levels). Default: 1')
parser.add_argument('--weight_path', type=str, help='path to the weight of the unet model')

# data
parser.add_argument('--save_path', type=str, help='path to a directory to save the features')
parser.add_argument('--dataset_name', default="brats", help='dataset to use')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloaders.')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs. 800 per default because Manon said it')
parser.add_argument('--root_dir', type=str, default='/data/brats')
parser.add_argument('--skip_training', default=False, action='store_true')

args = parser.parse_args()


class FeaturesDataset(Dataset):
    def __init__(self, root_dir, level):
        """
        Args:
            csv_file (string): name of the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.level = level
        self.root_dir = root_dir

    def __len__(self):
        return len(glob.glob(f'{self.root_dir}/features/*/level{self.level}.npy'))

    def __getitem__(self, idx):
        features_file = os.path.join(self.root_dir, "features", "{:08d}".format(idx),
                                     "level" + str(self.level) + ".npy")
        features_npy = np.load(features_file, allow_pickle=True)
        features = torch.tensor(features_npy, dtype=torch.float32)
        return features


def get_density_dataloader(args, root_dir, level):
    dataset = FeaturesDataset(root_dir, level)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True,
                        pin_memory=True)
    return loader


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):
    # torch.use_deterministic_algorithms(False)
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    set_determinism(seed=seed_val)

    # create saves dir if not exist :
    save_dir = f'/{args.save_path}/{args.name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    ''' Create dataset '''

    if args.dataset_name == 'brats':

        root_dir = args.root_dir

        if args.prepare:
            batch_size = 1
            roi_size = (96, 96, 96)
            model = get_model_brats(device, (4,) + roi_size, name='UNet', constrained=True, in_channels=4,
                                    out_channels=4,
                                    inference=True)
            model.load_state_dict(torch.load(args.weight_path))
            loader_images, _, _ = get_dataset_brats(batch_size,
                                                    roi_size,
                                                    root_dir,
                                                    args.num_workers,
                                                    args.seed,
                                                    train_with_val_transform=True)

            root_dir_features = r"/data"
            if not os.path.exists(os.path.join(root_dir_features, "features")):
                os.makedirs(os.path.join(root_dir_features, "features"))

            prepare_features_brats(model,
                                   loader_images,
                                   root_dir_features,
                                   device)

    channel_list = [320, 160, 80, 40, 20]  # level1 ... level5
    fuse = True
    for level in args.levels:
        ''' Initialize dataloaders '''
        loader = get_density_dataloader(args, root_dir, level)  # TO DO : BATCH SIZE !!

        ''' Prepare wandb run '''
        wandb.login()
        wandb.init(project=args.wandb_project, entity='pilabopoulos')
        wandb.run.name = f'{args.name}_level{level}'

        ''' Initialize the model '''

        epoch_num = args.n_epochs
        lowest_loss = 1e35
        epoch_loss_values, metric_values = [], []

        # use amp to accelerate training
        # scaler = torch.cuda.amp.GradScaler()
        ''' Init model with first batch '''
        for batch_idx, inputs in enumerate(
                loader):  # TO DO: les FMs sont déjà pré-enregistrées par batch?
            with torch.no_grad():
                inputs = inputs.squeeze(dim=0).transpose(0, 1).flatten(start_dim=1).transpose(0,
                                                                                              1).to(
                    'cpu')  # BxCx... -> CxBx... -> Cx... -> ...xC
                model = GMMv2(num_channels=channel_list[level - 1], x=inputs).to(device)
                inputs = inputs.to(device)
                outputs = - model(inputs.to(device))
                loss = outputs.mean()
                lowest_loss = loss.item()
                torch.save(model.state_dict(), os.path.join(save_dir, f'GMM_level{level}.pth'))
                print("init loss is:", loss.item())
                print("saved new best metric model")

            break
        if not args.skip_training:
            optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                         weight_decay=0.0)  # TO DO: quel est-ce qu'on doit faire un try catch sur lr ?

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

            ''' Training loop '''
            for epoch in range(epoch_num):
                print("-" * 10)
                print(f"epoch {epoch + 1}/{epoch_num}")
                model.train()
                epoch_loss = 0
                step = 0
                for batch_idx, inputs in enumerate(
                        loader):  # TO DO: les FMs sont déjà pré-enregistrées par batch?

                    inputs = inputs.squeeze(dim=0).transpose(0, 1).flatten(start_dim=1).transpose(0,
                                                                                                  1)  # BxCx... -> CxBx... -> Cx... -> ...xC
                    inputs = inputs.to(device)
                    '''
                    if not model.initialized:
                        model.match_params_kmeans(inputs)
                        model.eval()
                        outputs = - model(inputs)
                        loss = outputs.mean()
                        lowest_loss = loss.item()
                        torch.save(model.state_dict(), os.path.join(save_dir, f'GMM_level{level}.pth'))
                        print("init loss is:", loss.item())
                        print("saved new best metric model")
                        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                                     weight_decay=0.0)  # TO DO: quel est-ce qu'on doit faire un try catch sur lr ?
    
                        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
                        model.train()
                    '''
                    optimizer.zero_grad()
                    # with torch.cuda.amp.autocast():
                    outputs = - model(inputs)
                    if fuse:
                        fuse = False
                        graph = make_dot(outputs, params=dict(model.named_parameters()))

                        graph.format = 'png'
                        graph.render('graph')
                        #wandb.log({'Computation graph': wandb.Image(graph)})
                    loss = outputs.mean()
                    # scaler.scale(loss).backward()
                    loss.backward()
                    epoch_loss += loss.item()
                    print('loss', loss.item())
                    # scaler.step(optimizer)
                    optimizer.step()
                    model.clamp_L_diag()  # avoid unstable covariance matrix

                    step += 1

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)

                print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

                lr_scheduler.step()
                current_lr = lr_scheduler.get_last_lr()
                wandb.log(
                    {'Negative log-likelihood': epoch_loss,
                     'Learning rate': current_lr},
                    step=epoch)
                if epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss

                    torch.save(model.state_dict(), os.path.join(save_dir, f'GMM_level{level}.pth'))
                    print("saved new best metric model")

        wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
