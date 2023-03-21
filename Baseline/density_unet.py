"""
@author: gerinb
"""

import argparse
import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import faiss
from unet import *
from tqdm import tqdm


class DensityUnet(nn.Module):
    def __init__(self, path_density_unet, path_gmms, unet, device, combination='last', K=4):
        super(DensityUnet, self).__init__()
        self.Unet = unet
        self.Unet_init = False
        self.device = device
        self.GMM1 = None
        self.GMM2 = None
        self.GMM3 = None
        self.GMM4 = None
        self.GMM5 = None
        self.GMM1_init = False
        self.GMM2_init = False
        self.GMM3_init = False
        self.GMM4_init = False
        self.GMM5_init = False

        self.combination = combination

        self.channel_list = [20, 40, 80, 160, 320]

        
        if path_density_unet is not None:
            self._load_unet(path_density_unet)
        

        if path_gmms is not None:
            self._load_GMMS(path_gmms, K=K)

    def _load_unet(self, path):
        self.Unet.load_state_dict(torch.load(path, map_location=self.device))
        self.Unet_init = True

    def _load_GMMS(self, path, K=4):
        if path[0] != '_':
            self.GMM1 = GMMv1(num_channels=self.channel_list[4], K=K, init=False)
            self.GMM1.load_state_dict(torch.load(path[0], map_location=self.device))
            self.GMM1_init = True
        if path[1] != '_':
            self.GMM2 = GMMv1(num_channels=self.channel_list[3], K=K, init=False)
            self.GMM2.load_state_dict(torch.load(path[1], map_location=self.device))
            self.GMM2_init = True
        if path[2] != '_':
            self.GMM3 = GMMv1(num_channels=self.channel_list[2], K=K, init=False)
            self.GMM3.load_state_dict(torch.load(path[2], map_location=self.device))
            self.GMM3_init = True
        if path[3] != '_':
            self.GMM4 = GMMv1(num_channels=self.channel_list[1], K=K, init=False)
            self.GMM4.load_state_dict(torch.load(path[3], map_location=self.device))
            self.GMM4_init = True
        if path[4] != '_':
            self.GMM5 = GMMv1(num_channels=self.channel_list[0], K=K, init=False)
            self.GMM5.load_state_dict(torch.load(path[4], map_location=self.device))
            self.GMM5_init = True

    def forward(self, x):
        if self.combination == 'last':
            return self.combination_last(x)
        else:
            return None

    def forward_GMMS(self, x):
        density1 = self.GMM1(x[0])
        density2 = self.GMM2(x[1])
        density3 = self.GMM3(x[2])
        density4 = self.GMM4(x[3])
        density5 = self.GMM5(x[4])
        return [density1, density2, density3, density4, density5]

    def combination_last(self, x):

        B, C, H, W, Z = x.shape
        fms, seg = self.Unet.forward_fm_and_seg(x)

        if self.GMM5_init:
            fm5_flat = fms[-1].transpose(0, 1).flatten(start_dim=1).transpose(0, 1)
        else:
            raise Exception
        conf = self.GMM5(fm5_flat).view(B, 1, H, W, Z)
        return seg, conf

    def combination1(self, x):
        pass

    def combination2(self, x):
        pass

    # ... further combinations


def diag_elem(N):
    indices = torch.zeros(N, dtype=int)
    for i in range(1, N):
        indices[i] = indices[i - 1] + i + 1
    return indices


class GMMv2(nn.Module):
    def __init__(self, num_channels, K=None, x=None, max_k=5, rank=10, init_var=1e-3):
        super(GMMv2, self).__init__()
        clusters, cov, prop, best_k = self.init_centroids(x, max_k=max_k)

        self.rank = rank
        self.N = num_channels
        self.K = best_k
        self.weights = Parameter(torch.ones(self.K, ) / self.K)
        self.weights.data = prop
        self.loc = Parameter(torch.zeros(self.N, self.K))
        self.cov_factor = Parameter(torch.ones(self.N, self.rank, self.K) * init_var)
        self.cov_diag = Parameter(torch.ones(self.N, self.K))

        for k in range(self.K):
            self.loc.data[:, k] = torch.from_numpy(clusters[k])
            self.cov_diag.data[:, k] = torch.diag(cov[k])

        self.initialized = True

    def forward(self, x):

        B = x.shape[0]
        frac = 64 ** 3

        # use the log-sum-exp trick
        log_w = torch.log(nn.functional.softmax(self.weights, dim=0))
        lst = []

        for k in range(self.K):
            # print(self.loc[:, k].size(),self.cov_factor[:, :, k].size(), self.cov_diag[:, k].size())
            distribution = D.LowRankMultivariateNormal(self.loc[:, k], cov_factor=self.cov_factor[:, :, k],
                                                       cov_diag=self.cov_diag[:, k])  # .expand(torch.Size([frac]))
            # log_values = distribution.log_prob(x)
            log_values = torch.zeros(B).to(x.device)  # distribution.log_prob(x)
            for i in range(0, B, frac):
                sub_x = x[i:i + frac, :]
                log_values[i:i + frac] = distribution.log_prob(sub_x)
            if B % frac != 0:
                sub_x = x[(B // frac) * frac:, :]
                log_values[(B // frac) * frac:] = distribution.log_prob(sub_x)
            lst.append(log_values)

        log_p = torch.stack(lst, dim=0)
        log_sum = log_p + log_w.unsqueeze(dim=1)
        beta, _ = torch.max(log_sum, dim=0)
        output = beta + torch.logsumexp(log_sum - beta, dim=0)
        return output

    def init_centroids(self, x, max_k=5):
        niter = 20
        max_iter = 2
        verbose = False
        d = x.shape[1]
        k_values = range(max_k - 1, max_k)  # you can adjust the range of possible k values
        silhouette_scores = []
        for k in k_values:
            kmeans = faiss.Kmeans(d, k, niter=niter, verbose=verbose)
            print(f'K={k}')
            for iter in tqdm(range(max_iter)):
                idx = torch.randperm(x.shape[0])
                x_shuff = x[idx]
                kmeans.train(x_shuff)
            centroids = kmeans.centroids
            D, I = kmeans.index.search(x, 1)
            arr = np.array([np.sum(I == i) for i in range(k)])
            if np.sum(arr == 0) == 0:
                silhouette_scores.append(k)
                '''
                silhouette_sum = 0
                for i in range(k):
                    cluster_points = x[I.flatten() == i]
                    a_i = np.mean([np.linalg.norm(x - cluster_points[j], axis=1).mean()
                                   for j in range(cluster_points.shape[0])])
                    b_i = np.min([np.linalg.norm(x - centroids[j], axis=1).mean()
                                  for j in range(k) if j != i])
                    silhouette_sum += (b_i - a_i) / max(a_i, b_i)
                silhouette_scores.append(silhouette_sum / k)
                '''
            else:
                silhouette_scores.append(-100000)
        best_k = k_values[np.argmax(silhouette_scores)]
        print(f'Best K is {best_k}')
        kmeans = faiss.Kmeans(d, best_k, niter=niter, verbose=verbose)
        for iter in range(max_iter):
            idx = torch.randperm(x.shape[0])
            x_shuff = x[idx]
            kmeans.train(x_shuff)
        centroids = kmeans.centroids
        covs = []
        D, I = kmeans.index.search(x, 1)
        prop = torch.zeros(best_k)
        total_size = x.shape[0]
        for i in range(best_k):
            cluster_points = x[I.flatten() == i]
            cov = torch.cov(cluster_points.T)
            cov = cov + torch.eye(d) * 1e-9  # security margin
            covs.append(cov)
            prop[i] = cluster_points.shape[0] / total_size
        return centroids, covs, prop, best_k

    def clamp_L_diag(self, eps=1e-6):
        with torch.no_grad():
            # self.cov_diag.data = self.cov_diag.data.clamp_(-eps, None)
            # self.cov_factor.data = self.cov_factor.data.clamp_(eps,None)
            pass


class GMMv1(nn.Module):
    def __init__(self, num_channels, clusters=None, cov=None, prop=None, K=4, init=True):
        super(GMMv1, self).__init__()
        self.N = num_channels
        self.K = K
        self.tri_numel = int((self.N ** 2 - self.N) / 2 + self.N)
        self.tril_ind = torch.tril_indices(self.N, self.N, 0)
        self.diag_ind = diag_elem(self.N)
        self.weights = Parameter(torch.ones(self.K, ) / self.K)
        self.loc = Parameter(torch.zeros(self.N, self.K))
        self.scale_params = Parameter(torch.zeros(self.tri_numel, self.K))
        if init:
            self.weights.data = prop
            for k in range(self.K):
                self.loc.data[:, k] = torch.from_numpy(clusters[k])
                L = torch.linalg.cholesky(cov[k])
                self.scale_params.data[:, k] = L.float()[self.tril_ind[0], self.tril_ind[1]]

            self.initialized = True
        else:
            self.initialized = False

    def forward(self, x):

        # To avoid the error "RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling
        # `cublasStrsmBatched", we need to specify the batch_size in loc and scale :(
        # https://discuss.pytorch.org/t/how-to-use-torch-distributions-multivariate-normal-multivariatenormal-in
        # -multi-gpu-mode/135030/2)
        # x = x[0:100,:]
        B = x.shape[0]
        frac = 32 ** 3
        # use the log-sum-exp trick
        log_w = torch.log(nn.functional.softmax(self.weights, dim=0))
        lst = []
        for k in range(self.K):
            scale = torch.zeros(self.N, self.N).to(x.device)
            scale[self.tril_ind[0], self.tril_ind[1]] = self.scale_params[:, k]
            # detect invalid lower-cholesky (diag elements should be strictly positive)
            if torch.sum(self.scale_params[self.diag_ind, k] < 0):
                print('WARNING: invalid lower-cholesky matrix detected (diag elements should be strictly positive)')
            distribution = D.MultivariateNormal(self.loc[:, k], scale_tril=scale)  # .expand(torch.Size([B]))
            log_values = torch.zeros(B).to(x.device)  # distribution.log_prob(x)
            for i in range(0, B, frac):
                sub_x = x[i:i + frac, :]
                log_values[i:i + frac] = distribution.log_prob(sub_x)
            if B % frac != 0:
                sub_x = x[(B // frac) * frac:, :]
                log_values[(B // frac) * frac:] = distribution.log_prob(sub_x)
            lst.append(log_values)

        log_p = torch.stack(lst, dim=0)
        log_sum = log_p + log_w.unsqueeze(dim=1)
        beta, _ = torch.max(log_sum, dim=0)
        output = beta + torch.logsumexp(log_sum - beta, dim=0)
        return output

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.tril_ind = self.tril_ind.to(*args, **kwargs)
        return self

    def clamp_L_diag(self, eps=1e-2, maxx=None):
        with torch.no_grad():
            for k in range(self.K):
                self.scale_params[self.diag_ind, k] = self.scale_params[self.diag_ind, k].data.clamp_(eps, maxx)


def main():
    def test(input):
        return torch.max(input, dim=1, keepdim=True)

    x = torch.randn(1, 2, 3, 3, 3)
    print(x)
    y = x.transpose(0, 1).flatten(start_dim=1).transpose(0, 1)
    print(y)
    y, _ = test(y)
    print(y)
    x = y.view(1, 1, 3, 3, 3)

    print(x)

    exit()

    N = 10
    diag_ind = diag_elem(N)
    tril_ind = torch.tril_indices(N, N, 0)
    tri_numel = int((N ** 2 - N) / 2 + N)
    scale_params = torch.zeros((tri_numel))
    scale_params[diag_ind] = 1
    print(scale_params)
    scale = torch.zeros(N, N)

    scale[tril_ind[0], tril_ind[1]] = scale_params

    print(scale)
    print(scale[tril_ind[0], tril_ind[1]])
    exit()

    C = 320
    K = 2

    # x = torch.rand((100, 2))
    mean = np.ones((C,))
    cov = np.eye(C) / C
    x1 = torch.from_numpy(np.random.multivariate_normal(mean, cov, 1000))
    mean2 = np.ones((C,)) * 10
    # cov = np.eye(C)
    x2 = torch.from_numpy(np.random.multivariate_normal(mean2, cov, 500))
    x = torch.cat([x1, x2])
    print(x.shape)

    model = GMMv1(C, K, x)

    max_epochs = 10000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        out = - model(x)
        loss = out.mean()
        loss.backward(retain_graph=True)
        optimizer.step()
        lr_scheduler.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    print("Final parameters:")
    # print("loc:", model.loc.detach().numpy())

    for k in range(K):
        L = torch.zeros(model.N, model.N)
        L[model.tril_ind[0], model.tril_ind[1]] = model.scale_params[:, k]
        scale = L @ L.T
        print("scale:", scale.detach().numpy())

    print(nn.functional.softmax(model.weights, dim=0))


# %%
if __name__ == "__main__":
    main()
