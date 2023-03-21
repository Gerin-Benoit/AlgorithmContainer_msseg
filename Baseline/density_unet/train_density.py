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
from tqdm import tqdm
import glob

parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# login
parser.add_argument('--wandb_project', type=str, default='density', help='wandb project name')
parser.add_argument('--name', default="idiot without a name", help='Wandb run name')

# architecture
parser.add_argument('--levels', nargs='+', type=int, default=[5],
                    help='list of the levels to fit a GMM (pass 1 2 3 4 5 for all levels')
parser.add_argument('--weight_path', type=str, help='path to the weight of the unet model')

# data
parser.add_argument('--save_path', type=str, help='path to a directory to save the features')
parser.add_argument('--dataset_name', default="brats", help='dataset to use')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloaders.')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=150, help='Number of epochs. 800 per default because Manon said it')
parser.add_argument('--root_dir', type=str, default='/data/brats')
parser.add_argument('--skip_training', default=False, action='store_true')
parser.add_argument('--cache_rate', type=float, default=1.0)
parser.add_argument('--unconstrained', default=False, action='store_true')

# kmean and gmm params
parser.add_argument('--num_components', type=int, default=4, help='number of multivariate Gaussian in the mixture.')
parser.add_argument('--kmean_niter', type=int, default=20, help='number of iterations to fit the kmean for each batch')

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


def compute_observed_distribution_per_sample(x, y, prior):
    size = y.size(-1) // x.size(-1)  # compute spatial ratios
    B, C, H, W, Z = y.shape

    # Unfold the tensor into patches of size (C, a, a, a)
    out = torch.zeros(B, C, H // size, W // size, Z // size)
    unfolded = y.unfold(2, size, size).unfold(3, size, size).unfold(4, size, size)
    for c in range(C):
        out[:, c, :, :, :] = torch.sum(unfolded[:, c, :, :, :, :, :, :] == 1, dim=(-3, -2, -1))

    out = (out / prior.reshape(1, C, 1, 1, 1)).sum(dim=1)
    return out


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
        from datasets.brats.data import get_dataset as get_dataset_brats
        from datasets.brats.model import get_model as get_model_brats
        NUM_CLASSES = 4
        root_dir = args.root_dir

        batch_size = 1
        roi_size = (96, 96, 96)
        constrained = not args.unconstrained
        model = get_model_brats(device, (4,) + roi_size, name='UNet', constrained=constrained, in_channels=4,
                                out_channels=4,
                                inference=True)
        x = torch.randn((1, 4,) + roi_size).to(device)
        y = model(x)
        model.load_state_dict(torch.load(args.weight_path))
        model = model.to(device)
        loader_images, _, _ = get_dataset_brats(batch_size,
                                                roi_size,
                                                root_dir,
                                                args.num_workers,
                                                args.cache_rate,
                                                args.seed)
    if args.dataset_name == 'msseg':
        from datasets.msseg.data import get_dataset as get_dataset_msseg
        from datasets.msseg.model import get_model as get_model_msseg
        NUM_CLASSES = 2
        root_dir = args.root_dir

        batch_size = 1
        roi_size = (96, 96, 96)
        constrained = not args.unconstrained
        model = get_model_msseg(device, (1,) + roi_size, name='UNet', constrained=constrained, in_channels=1,
                                out_channels=2,
                                inference=True)
        x = torch.randn((1, 1,) + roi_size).to(device)
        y = model(x)
        model.load_state_dict(torch.load(args.weight_path))
        model = model.to(device)
        loader_images, _, _ = get_dataset_msseg(batch_size,
                                                roi_size,
                                                root_dir,
                                                args.num_workers,
                                                args.cache_rate,
                                                args.seed)

    """ Compute priors over the whole dataset """
    priors = torch.zeros(NUM_CLASSES)
    n_voxels = 0
    for batch_idx, batch_data in tqdm(enumerate(loader_images)):

        labels = batch_data["label"]
        n_voxels += torch.numel(labels[:, 0, :])
        for c in range(NUM_CLASSES):
            priors[c] += torch.sum(labels[:, c, :] == 1)
    priors /= n_voxels

    channel_list = [320, 160, 80, 40, 20]  # level1 ... level5
    for level in args.levels:

        ''' Prepare wandb run '''
        wandb.login()
        run = wandb.init(project=args.wandb_project, entity='pilabopoulos')
        wandb.run.name = f'{args.name}_level{level}'

        ''' Initialize the model '''

        epoch_num = args.n_epochs
        lowest_loss = 1e35
        epoch_loss_values, metric_values = [], []
        # use amp to accelerate training
        scaler = torch.cuda.amp.GradScaler()

        ''' *** START Init model over the dataset *** '''
        K = args.num_components
        kmeans = faiss.Kmeans(channel_list[level - 1], k=K, niter=args.kmean_niter, verbose=False)
        model.eval()
        weight_max = 0.0
        for batch_idx, batch_data in tqdm(enumerate(loader_images)):
            with torch.no_grad():
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"]
                fms = model.forward_fm(inputs)
                fm = fms[level - 1].to('cpu')
            weights = compute_observed_distribution_per_sample(fm, labels, priors)
            weights = weights.flatten()

            inputs = fm.transpose(0, 1).flatten(start_dim=1).transpose(0, 1)  # BxCx... -> CxBx... -> Cx... -> ...xC
            idx = torch.randperm(inputs.shape[0])
            x_shuff = inputs[idx]
            weights_shuff = weights[idx]
            kmeans.train(x_shuff, weights=weights_shuff)
            if torch.nanmean(weights) > weight_max:
                weight_max = torch.nanmean(weights)
                x_cov = x_shuff.detach()

        centroids = kmeans.centroids
        covs = []
        D, I = kmeans.index.search(x_cov, 1)
        prop = torch.zeros(K)
        total_size = x_cov.shape[0]
        for i in range(K):
            cluster_points = x_cov[I.flatten() == i]
            cov = torch.cov(cluster_points.T)
            cov = cov + torch.eye(channel_list[level - 1]) * 1e-6  # security margin
            covs.append(cov)
            prop[i] = cluster_points.shape[0] / total_size

        ''' *** END init model over the dataset *** '''

        gmm = GMMv1(num_channels=channel_list[level - 1], clusters=centroids, cov=covs, prop=prop, K=K).to(device)

        if not args.skip_training:
            optimizer = torch.optim.Adam(gmm.parameters(), args.lr, weight_decay=0.0)

            ''' Training loop '''
            for epoch in range(epoch_num):
                print("-" * 10)
                print(f"epoch {epoch + 1}/{epoch_num}")
                model.eval()
                gmm.train()
                epoch_loss = 0
                epoch_loss_K = torch.zeros(NUM_CLASSES)
                step = 0
                for batch_idx, batch_data in enumerate(loader_images):
                    with torch.no_grad():
                        inputs = batch_data["image"].to(device)
                        labels = batch_data["label"]
                        fms = model.forward_fm(inputs)
                        fm = fms[level - 1].to('cpu')
                    inputs = fm.transpose(0, 1).flatten(start_dim=1).transpose(0,
                                                                               1)  # BxCx... -> CxBx... -> Cx... -> ...xC
                    weights = compute_observed_distribution_per_sample(fm, labels, priors)
                    weights = weights.flatten().to(device)
                    inputs = inputs.to(device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = - gmm(inputs)
                        loss = torch.nansum(outputs * weights) / torch.sum(weights)

                    if level == 5:
                        labels = labels.transpose(0, 1).flatten(start_dim=1).to(device)
                        loss_K = torch.zeros(NUM_CLASSES)
                        for k in range(NUM_CLASSES):
                            loss_k = outputs[labels[k] == 1].nanmean()
                            loss_K[k] = loss_k.item()
                        epoch_loss_K += torch.nan_to_num(loss_K, nan=0)

                    scaler.scale(loss).backward()
                    epoch_loss += loss.item()

                    if level == 5:
                        print('loss', loss.item(), '  per class: ', loss_K)
                    else:
                        print('loss', loss.item())

                    scaler.step(optimizer)
                    scaler.update()

                    gmm.clamp_L_diag()  # avoid unstable covariance matrix

                    step += 1

                epoch_loss /= step
                epoch_loss_K /= step
                epoch_loss_values.append(epoch_loss)

                if level == 5:
                    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f},   per class: {epoch_loss_K}")
                else:
                    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

                wandb.log({'Negative log-likelihood': epoch_loss}, step=epoch)

                if level == 5:
                    for k in range(NUM_CLASSES):
                        wandb.log({f'Negative log-likelihood of class {k}': epoch_loss_K[k]}, step=epoch)
                if epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss

                    torch.save(gmm.state_dict(), os.path.join(save_dir, f'GMM_level{level}.pth'))
                    print("saved new best metric model")

        run.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
