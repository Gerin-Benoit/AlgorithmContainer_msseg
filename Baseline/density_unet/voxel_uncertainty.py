"""
Source : https://github.com/NataliiaMolch/MS_WML_uncs/blob/main/voxel_uncertainty_measures.py
"""
import torch
def entropy_of_expected(probs, epsilon=1e-10):
    mean_probs = torch.mean(probs, axis=0)
    log_probs = -torch.log(mean_probs + epsilon)
    return torch.sum(mean_probs * log_probs, axis=-1)


def expected_entropy(probs, epsilon=1e-10):
    log_probs = -torch.log(probs + epsilon)
    return torch.mean(torch.sum(probs * log_probs, axis=-1), axis=0)

def mean_meanlog(probs, epsilon=1e-10):
    mean_probs = torch.mean(probs, axis=0)
    mean_lprobs = torch.mean(torch.log(probs + epsilon), axis=0)
    return torch.sum(mean_probs*mean_lprobs, axis=-1)

def variance(probs):
    var_probs = torch.var(probs, axis=0)
    return var_probs
    

def voxel_uncertainty(probs, measure='MI'):
    
    if measure == 'MI': # mutual information => epistemic uncertainty
        eoe = entropy_of_expected(probs)
        #print('eoe',eoe)
        exe = expected_entropy(probs)
        #print("exe",exe)
        unc = eoe - exe 
    elif measure == 'EOE': # entropy of expected => total uncertainty
        unc = entropy_of_expected(probs)
    elif measure == 'RMI':
        exe = expected_entropy(probs)
        eoe = entropy_of_expected(probs)
        unc = - mean_meanlog(probs) - eoe
    elif measure == 'variance':
        unc = 1-variance(probs)
    return unc

if __name__ == '__main__':
    p = torch.rand((3,4,4,4, 2)) # (num_models, H, W, Z, num_classes)
    unc = voxel_uncertainty(p) # (H,W,Z)
    print("probs shape",p.size(),'unc shape',unc.size())  
    