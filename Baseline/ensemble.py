import torch
from torch import nn
from voxel_uncertainty import voxel_uncertainty

class EnsembleUnet(nn.Module):
    def __init__(self, Unets, paths=None, uncertainty='MI', reduction='mean', act=nn.Softmax(dim=1)):
        super(EnsembleUnet, self).__init__()
        self.Unets = Unets # list of Unets
        self.act = act
        self.reduction= reduction
        self.Unets_init = False
        if paths is not None:
            self._load_unets(paths)
        self.uncertainty = uncertainty
        self.n_models = len(paths)
        
        
    def get_model(self, index):
        return self.Unets[index]
    
    def combine_outputs(self, outputs):
        if self.reduction == 'mean':
            return torch.stack(outputs).mean(axis=0)
        elif self.reduction == 'majority_vote':
            mode, counts = torch.mode(torch.argmax(torch.stack(outputs), axis=2, keepdim=True), dim=0)
            one_hot_output = nn.functional.one_hot(mode.squeeze(1)).permute(0,4,1,2,3)

            #majority[torch.arange(majority.shape[0]), mode.argmax(dim=1)] = 1
            # Return the majority one-hot vector as the output
            return one_hot_output
            
        
    def forward(self, x):
        outputs = []
        for i in range(len(self.Unets)):
            outputs.append(self.act(self.Unets[i](x)))
            
        return final_outputs, -uncertainty_map.unsqueeze(1)
    
    def get_uncertainty(self, outputs):
        # outputs shape list of num_models tensors of shape (B,num_classes,H,W,Z)
        # need tensor of shape (num_models, B, H, W, Z, num_classes)
        outputs = torch.stack(outputs).transpose(2,-1)
        unc_map = voxel_uncertainty(outputs, measure=self.uncertainty)
        return unc_map
    
    def _load_unets(self, paths):
        assert len(paths) == len(self.Unets)
        for i in range(len(paths)):
            self.Unets[i].load_state_dict(torch.load(paths[i]))
        self.Unets_init = False


    def entropy(self, x):
        pass

    # ... further combinations
    
