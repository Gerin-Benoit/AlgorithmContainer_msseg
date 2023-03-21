"""
adapted from https://github.com/Shifts-Project/shifts/tree/main/mswml
"""

import argparse
import os
import torch
from monai.data import decollate_batch
from monai.transforms import Compose, AsDiscrete
from torch import nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, GeneralizedDiceLoss
import numpy as np
import random
from data_load import get_train_dataloader, get_val_dataloader, get_flair_dataloader
from model import get_model
from os.path import join as pjoin
import wandb
from monai.utils import set_determinism
import math
from monai.metrics import DiceMetric

from copy import deepcopy
from spectral_norm_conv_inplace import remove_spectral_norm_conv
from spectral_norm_fc import remove_spectral_norm


parser = argparse.ArgumentParser(description='Get all command line arguments.')
# training
parser.add_argument('--lr', type=float, default=1e-3, help='Specify the initial learning rate')

parser.add_argument('--n_epochs', type=int, default=300, help='Specify the number of epochs to train for')
# initialisation
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
# data
parser.add_argument('--data_dir', type=str, required=True, help='Specify the path to the data files directory')
parser.add_argument('--save_path', type=str, required=True, default='/saves', help='Specify the path to the save directory')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
# logging
parser.add_argument('--val_interval', type=int, default=5, help='Validation every n-th epochs')

parser.add_argument('--constrained', type=bool, default=False, help='Whether to use a constrained or unconstrained unet')

parser.add_argument('--wandb_project', type=str, default='shift_ms', help='wandb project name')
parser.add_argument('--name', default="idiot without a name", help='Wandb run name')
parser.add_argument('--focal', action='store_true', default=False)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--cache_rate', default=1.0, type=float)



VAL_AMP = True
roi_size = (96, 96, 96)

def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi_size,
            sw_batch_size=6,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


post_trans = Compose(
    [AsDiscrete(argmax=True, to_onehot=2)]
)  # adaptation for softmax activation and 2 classes (1+background)

def main(args):
    #torch.use_deterministic_algorithms(False)
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

    training_flair_path = pjoin(args.data_dir, "msseg", "unsupervised", "flair")

    '''' Initialise dataloaders '''
    train_loader = get_flair_dataloader(flair_path=training_flair_path,
                                        num_workers=args.num_workers,
                                        seed=seed_val,
                                        cache_rate=args.cache_rate,
                                        ssl=True)

    ''' Initialise the model '''
    model = get_model(device, (1,) + roi_size, 'UNet', constrained=args.constrained, in_channels=1, out_channels=1)


    loss_function = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)


    epoch_num = args.n_epochs
    val_interval = args.val_interval
    sw_batch_size = 4
    gamma_focal = 2.0
    best_metric, best_metric_epoch = -1, -1
    epoch_loss_values, metric_values = [], []
    # dice_metric = DiceMetric(include_background=False, reduction="mean")
    val_dice_metric = DiceMetric(include_background=False, reduction="mean")

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()

    torch.backends.cudnn.benchmark = True

    ''' Training loop '''
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        epoch_loss_ce = 0
        epoch_loss_dice = 0
        step = 0
        for batch_idx, batch_data in enumerate(train_loader):
            n_samples = batch_data["image"].size(0)
            for m in range(0, batch_data["image"].size(0), args.batch_size):

                step += args.batch_size
                inputs = batch_data["image"][m:(m + args.batch_size)].to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, inputs)


                scaler.scale(loss).backward()
                #loss.backward()

                epoch_loss += loss.item()

                #print('loss',loss.item(),'loss_ce', loss_ce.item(), 'loss_dice', loss_dice.item())


                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model.clamp_norm_layers()

                if step % 100 == 0:
                    step_print = int(step / args.batch_size)
                    print(
                        f"{step_print}/{(len(train_loader) * n_samples) // (train_loader.batch_size * 2)}, train_loss: {loss.item():.4f}")

            #current_lr = adjust_learning_rate(optimizer, batch_idx/len(train_loader)+epoch, args)
            #print(current_lr)

            # outputs = [post_trans(i) for i in decollate_batch(outputs)]
            # dice_metric(y_pred=outputs, y=labels)

        # metric = dice_metric.aggregate().item()
        epoch_loss /= step_print

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # dice_metric.reset()

        ''' Validation '''
        if (epoch + 1) % val_interval == 0:
            save_path = os.path.join(save_dir, f"best_metric_model_{args.name}_seed_{args.seed}.pth")
            model.eval()


            with torch.no_grad():
                for batch_idx, batch_data in enumerate(train_loader):

                    for m in range(0, batch_data["image"].size(0), args.batch_size):
                        inputs = batch_data["image"][m:(m + args.batch_size)].to(device)
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            #loss = loss_function(outputs, inputs)
                    break
                torch.cuda.empty_cache()




                if epoch_loss > best_metric:
                    best_metric = epoch_loss
                    best_metric_epoch = epoch + 1
                    name = "unconstrained" if args.constrained else "constrained"
                    if args.constrained:
                        model = model.cpu()
                        model_copy = deepcopy(model)
                        model = model.to(device)
                        for name, p in model_copy.named_modules():

                            if isinstance(p, nn.Conv3d) and not (p.kernel_size==(1,1,1) and p.out_channels==2):
                                remove_spectral_norm(p)

                        torch.save(
                            model_copy.state_dict(),
                            os.path.join(save_dir, f'pretrained_{name}_seed{seed_val}.pth'),
                        )
                    else:
                        torch.save(
                            model.state_dict(),
                            os.path.join(save_dir, f'pretrained_{name}_seed{seed_val}.pth'),
                        )
                    print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                      f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                      )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
