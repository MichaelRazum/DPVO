import torch
import cuda_ba

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject

def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations, eff_impl=False, alpha1=0., alpha2=0., c_depth_reg=False, depth_prior=None):
    if depth_prior is None:
        depth_prior = torch.zeros(ii.pg.ii.shape[0], device=patches.device, dtype=torch.float32)

    else:
        depth_prior = depth_prior.to(device=patches.device, dtype=torch.float32)
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, M, t0, t1, iterations, eff_impl, alpha1, alpha2, c_depth_reg, depth_prior)