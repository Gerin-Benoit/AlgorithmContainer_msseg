"""
Source : https://github.com/NataliiaMolch/MS_WML_uncs/blob/main/voxel_uncertainty_measures.py
"""
import torch


def entropy_of_expected(probs, mask=None, epsilon=1e-10):
    print("EOE")
    if mask == None:
        print("if")
        mask = torch.eq(probs, -1)  # create boolean mask for values equal to -1
        probs_filtered = probs.clone()  # make a copy of probs tensor
        probs_filtered[mask] = 0  # set -1 values to 0 in filtered tensor
        non_negative_count = torch.sum(~mask, axis=0)  # count non-negative values
        mean_probs = torch.sum(probs_filtered,
                               axis=0) / non_negative_count  # compute mean excluding -1 values (for example if only 2 confident models -> divide the sum only by 2)
        log_probs = torch.log(mean_probs + epsilon)

        eoe = -(mean_probs * log_probs)
        return eoe
    else:
        print("else")
        weighted_mask = mask.float() + (1 - mask.float()) * 0.5
        probs_filtered = probs.clone()  # make a copy of probs tensor
        non_negative_count = torch.sum(weighted_mask, axis=0)  # count non-negative values
        mean_probs = torch.sum(weighted_mask * probs_filtered,
                               axis=0) / non_negative_count  # compute mean excluding -1 values (for example if only 2 confident models -> divide the sum only by 2)
        log_probs = torch.log(mean_probs + epsilon)
        eoe = -(mean_probs * log_probs)
        return eoe


def entropy_of_expectedv2(predictions, epsilon=1e-10):
    #  predictions : M, H, W, Z (probs for class 1)
    # assert len(predictions.shape) == 4, "Input tensor must have shape (num_models, H, W, Z)"
    entropy = -torch.sum(predictions * torch.log(predictions + epsilon), dim=0)
    expectation_of_entropy = entropy / predictions.shape[0]
    return expectation_of_entropy  # minus to return a confidence measure


def expected_entropy(probs, epsilon=1e-10):
    log_probs = -torch.log(probs + epsilon)
    return torch.mean(torch.sum(probs * log_probs, axis=-1), axis=0)


def mean_meanlog(probs, epsilon=1e-10):
    mean_probs = torch.mean(probs, axis=0)
    mean_lprobs = torch.mean(torch.log(probs + epsilon), axis=0)
    return torch.sum(mean_probs * mean_lprobs, axis=-1)


def variance(probs):
    var_probs = torch.var(probs, axis=0)
    return var_probs


def voxel_uncertainty(probs, mask=None, measure='MI'):
    if measure == 'MI':  # mutual information => epistemic uncertainty
        eoe = entropy_of_expected(probs)
        # print('eoe',eoe)
        exe = expected_entropy(probs)
        # print("exe",exe)
        unc = eoe - exe
    elif measure == 'EOE':  # entropy of expected => total uncertainty
        unc = entropy_of_expected(probs, mask)
    elif measure == 'EOEv2':  # expectation of entropy => AU
        unc = entropy_of_expectedv2(probs)
    elif measure == 'RMI':
        exe = expected_entropy(probs)
        eoe = entropy_of_expected(probs)
        unc = - mean_meanlog(probs) - eoe
    elif measure == 'variance':
        unc = 1 - variance(probs)
    return unc


if __name__ == '__main__':
    p = torch.rand((3, 4, 4, 4, 2))  # (num_models, H, W, Z, num_classes)
    unc = voxel_uncertainty(p)  # (H,W,Z)
    print("probs shape", p.size(), 'unc shape', unc.size())
