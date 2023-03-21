"""
Build nDSC retention curve plot.
"""

import argparse
import os
import torch
import torch.nn as nn
from joblib import Parallel
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import numpy as np
from .metrics import multi_class_dsc_retention_curve
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set_theme()
from sklearn import metrics
from tqdm import tqdm

from scipy.stats import entropy

from monai.data import decollate_batch
from monai.metrics import DiceMetric

def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def compute_retention_curve(super_model,
                            roi_size,
                            test_loader,
                            post_trans,
                            save_path,
                            ensemble,
                            device):

    torch.multiprocessing.set_sharing_strategy('file_system')


    output_classes = 2
    VAL_AMP = True

    # define inference method
    def inference(input, model):
        def _compute(input, model):
            return sliding_window_inference(
                inputs=input,
                roi_size=roi_size,
                sw_batch_size=6,
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input, model)
        else:
            return _compute(input, model)

    # Significant class imbalance means it is important to use logspacing between values
    # so that it is more granular for the higher retention fractions
    fracs_retained = np.log(np.arange(200 + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)

    val_dice_metric = DiceMetric(include_background=False, reduction="mean")
    val_dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

    act=nn.Softmax(dim=1)

    ''' Evaluation loop '''

    with Parallel(n_jobs=8) as parallel_backend:
        with torch.no_grad():
            fracs_retained = np.log(np.arange(200 + 1)[1:])
            fracs_retained /= np.amax(fracs_retained)
            dsc_rc_curve = []
            dsc_per_class_rc_curve = []
            for val_data in tqdm(test_loader):
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device)
                )
                #val_outputs, val_confidences = model(val_inputs)
                #print(val_outputs.size(), val_confidences.size())
                if ensemble:
                    ens_outputs = []
                    for i in range(super_model.n_models):
                        model = super_model.get_model(i)
                        outputs = inference(val_inputs, model)
                        ens_outputs.append(act(outputs))
                    val_outputs = super_model.combine_outputs(ens_outputs)
                    val_confidences = - super_model.get_uncertainty(ens_outputs)
                else:
                    val_outputs, val_confidences = inference(val_inputs, super_model)

                val_outputs_trans = [post_trans(i) for i in decollate_batch(val_outputs)]

                val_dice_metric_batch(y_pred=val_outputs_trans, y=val_labels)
                val_dice_metric(y_pred=val_outputs_trans, y=val_labels)

                gt = torch.argmax(val_labels, axis=1).cpu().numpy()
                val_outputs = torch.argmax(val_outputs, axis=1).cpu().numpy()
                #seg = act(val_outputs).cpu().numpy()
                conf_map = val_confidences.cpu().numpy()

                #entropy_output = entropy([seg[:, 0, :], seg[:, 1, :]], base=2)
                #uncs_map_alea = np.copy(np.squeeze(entropy_output))  # or 1 - ... ?
                uncs_map = - conf_map
                #seg = np.squeeze(seg[0, 1])
                #seg[seg >= thresh] = 1
                #seg[seg < thresh] = 0
                #exit()

                dsc_scores, dsc_scores_per_class = multi_class_dsc_retention_curve(ground_truth=gt.flatten(),
                                                            predictions=val_outputs.flatten(),
                                                            uncertainties=uncs_map.flatten(),
                                                            fracs_retained=fracs_retained,
                                                            parallel_backend=parallel_backend)
                dsc_rc_curve += [dsc_scores]
                dsc_per_class_rc_curve += [dsc_scores_per_class]



            unc_values = np.mean(np.asarray(dsc_rc_curve), axis=0)
            unc_values_per_class = np.mean(np.asarray(dsc_per_class_rc_curve), axis=0)


    # Iterate through each entry in unc_values_per_class and generate a separate figure for each one
    for i in range(output_classes):
        fig = plt.figure(i)
        plt.plot(fracs_retained, unc_values_per_class[:,i], label=f"Class {i} DSC R-AUC : {1. - metrics.auc(fracs_retained, unc_values_per_class[:, i]):.6f}")
        plt.xlabel("Retention Fraction")
        plt.ylabel("DSC")
        plt.xlim([0.0, 1.01])
        plt.legend()
        plt.savefig(os.path.join(save_path, f"retention_curve_class_{i}.png"))
        plt.clf()

    plt.figure(0)
    plt.plot(fracs_retained, unc_values,
             label=f"DSC R-AUC : {1. - metrics.auc(fracs_retained, unc_values):.5f}")
    plt.xlabel("Retention Fraction")
    plt.ylabel("DSC")
    plt.xlim([0.0, 1.01])
    plt.legend()
    plt.savefig(os.path.join(save_path, f"retention_curve.png"))
    plt.clf()




