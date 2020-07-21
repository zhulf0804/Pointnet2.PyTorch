import torch
from utils.common import get_dists


def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long)
    dists = torch.ones(B, N) * 1e5
    inds = torch.randint(0, N, size=(B, )).long()
    batchlists = torch.arange(0, B)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz))
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids