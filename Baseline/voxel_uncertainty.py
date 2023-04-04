"""
Source : https://github.com/NataliiaMolch/MS_WML_uncs/blob/main/voxel_uncertainty_measures.py
"""
import torch
def entropy_of_expected(probs, mask=None, weight_unconfident=0.5, epsilon=1e-10):
    if mask == None:
        mask = torch.eq(probs, -1)  # create boolean mask for values equal to -1
        probs_filtered = probs.clone()  # make a copy of probs tensor
        probs_filtered[mask] = 0  # set -1 values to 0 in filtered tensor
        non_negative_count = torch.sum(~mask, axis=0)  # count non-negative values
        mean_probs = torch.sum(probs_filtered, axis=0) / non_negative_count  # compute mean excluding -1 values (for example if only 2 confident models -> divide the sum only by 2)
        log_probs = torch.log(mean_probs + epsilon)

        eoe = -(mean_probs * log_probs)
        return eoe
    else:
        weighted_mask = mask.float() + (1 - mask.float())*weight_unconfident
        probs_filtered = probs.clone()  # make a copy of probs tensor
        non_negative_count = torch.sum(weighted_mask, axis=0)  # count non-negative values
        mean_probs = torch.sum(weighted_mask * probs_filtered, axis=0) / non_negative_count  # compute mean excluding -1 values (for example if only 2 confident models -> divide the sum only by 2)
        log_probs = torch.log(mean_probs + epsilon)
        eoe = -(mean_probs * log_probs)
        return eoe


def expectation_of_entropy(predictions, probs, epsilon=1e-10):

    entropy = -torch.sum(predictions * torch.log(predictions+epsilon), dim=0)
    expectation_of_entropy = entropy / predictions.shape[0]
    return expectation_of_entropy

def weighted_expectation_of_entropy(probs, confs, epsilon=1e-10):
    """
    Computes the weighted expectation of entropy using the confidence scores.

    Args:
    probs: tensor of shape (M, H, W, Z) containing predictions of the models
    confs: tensor of shape (M, H, W, Z) containing confidence scores for each prediction
    epsilon: float, small value to avoid log(0)

    Returns:
    A tensor of shape (H, W, Z) containing the weighted expectation of entropy
    """
    # Compute the normalized weights for each model based on the confidence scores
    weights = confs / torch.sum(confs, dim=0, keepdim=True)

    # Compute the entropy for each model's predictions
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=0)

    # Compute the weighted expectation of entropy
    weighted_entropy = torch.sum(weights * entropy, dim=0)

    return weighted_entropy

def expected_entropy(probs, mask = None, weight_unconfident=0.5, epsilon=1e-10):
    if mask==None:
        log_probs = -torch.log(probs + epsilon)
        return torch.mean(torch.sum(probs * log_probs, axis=-1), axis=0)
    else:
        weighted_mask = mask.float() + (1 - mask.float())*weight_unconfident
        probs_filtered = probs.clone()  # make a copy of probs tensor
        non_negative_count = torch.sum(weighted_mask, axis=0)  # count non-negative values
        log_probs = -torch.log(probs + epsilon)
        exe = torch.sum(probs * log_probs * weighted_mask , axis=0)/non_negative_count
        return exe



def mean_meanlog(probs, mask=None, weight_unconfident=0.5, epsilon=1e-10):
    if mask == None:
        mean_probs = torch.mean(probs, axis=0)
        mean_lprobs = torch.mean(torch.log(probs + epsilon), axis=0)
        return torch.sum(mean_probs*mean_lprobs, axis=-1)
    else:
        weighted_mask = mask.float() + (1 - mask.float())*weight_unconfident
        probs_filtered = probs.clone()  # make a copy of probs tensor
        non_negative_count = torch.sum(weighted_mask, axis=0)  # count non-negative values
        mean_probs = torch.sum(probs * weighted_mask , axis=0)/non_negative_count
        mean_lprobs = torch.sum(torch.log(probs + epsilon)*weighted_mask, axis=0)/non_negative_count
        mml = torch.sum(mean_probs*mean_lprobs, axis=-1)
        return mml



def variance(probs):
    var_probs = torch.var(probs, axis=0)
    return var_probs


def voxel_uncertainty(probs, mask = None, measure='MI', weight_unconfident=0.5):

    if measure == 'MI': # mutual information => epistemic uncertainty
        eoe = entropy_of_expected(probs, weight_unconfident)
        #print('eoe',eoe)
        exe = expected_entropy(probs, weight_unconfident)
        #print("exe",exe)
        unc = eoe - exe
    elif measure == 'EXE':
        unc = expected_entropy(probs, mask, weight_unconfident)
    elif measure == 'EOE':  # entropy of expected => total uncertainty
        unc = entropy_of_expected(probs, mask, weight_unconfident)
    elif measure == 'EOEv2':  # expectation of entropy => AU
        unc = entropy_of_expectedv2(probs)
    elif measure == 'RMI':
        exe = expected_entropy(probs, mask, weight_unconfident)
        eoe = entropy_of_expected(probs, mask, weight_unconfident)

        unc = -(- mean_meanlog(probs) - eoe)
    elif measure == 'variance':
        unc = 1-variance(probs)
    return unc

if __name__ == '__main__':
    p = torch.rand((3,4,4,4, 2)) # (num_models, H, W, Z, num_classes)
    unc = voxel_uncertainty(p) # (H,W,Z)
    print("probs shape",p.size(),'unc shape',unc.size())
