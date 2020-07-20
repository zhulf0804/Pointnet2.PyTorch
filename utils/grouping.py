import torch
from utils.common import gather_points, get_dists


def ball_query(points, query_inds, radius, K):
    '''

    :param points: shape=(B, N, C)
    :param query_inds: shape=(B, M)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    B, N, C = points.shape
    M = query_inds.shape[1]
    group_inds = torch.arange(0, N).view(1, 1, N).repeat(B, M, 1)
    cur_points = gather_points(points, query_inds)
    dists = get_dists(cur_points, points)
    group_inds[dists > radius] = N
    group_inds = torch.sort(group_inds, dim=-1)[0][:, :, K]
    group_min_inds = group_inds[:, :, 0:1].repeat(1, 1, K)
    group_inds[group_inds == N] = group_min_inds[group_inds == N]
    return group_inds