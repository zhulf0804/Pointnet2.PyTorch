import torch
from utils.common import get_dists


def fps(points, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param points: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    B, N, C = points.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long)
    dists = torch.ones(B, N) * 1e5
    inds = torch.randint(0, N, size=(B, ))
    batchlists = torch.arange(0, B)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = points[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), points))
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids