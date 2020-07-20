import torch
import torch.nn as nn
from utils.sampling import fps
from utils.grouping import ball_query
from utils.common import gather_points


class PointNet(nn.Module):
    def __init__(self, in_channels, mlp):
        super(PointNet, self).__init__()
        self.block_list = []
        for i, out_channels in enumerate(mlp):
            self.block_list.append([nn.Conv2d(in_channels, out_channels, 1,
                                             stride=1, padding=0, bias=True),
                                    nn.ReLU(out_channels)])

    def forward(self, x):
        for conv, acti in self.block_list:
            x = acti(conv(x))
        x = torch.max(x, dim=2, keepdim=True)
        return x


def sample_and_group(points, M, radius, K):
    '''

    :param points: shape=(B, N, C)
    :param M:
    :param radius:
    :param K:
    :return: new_points, shape=(B, M, K, C)
    '''
    centroids = fps(points, M)
    group_inds = ball_query(points, centroids, radius, K)
    new_points = gather_points(points, group_inds)
    return new_points


def pointnet_sa_module(points, M, radius, K, mlp):
    pointnet_module = PointNet(6, mlp)
    new_points = sample_and_group(points, M, radius, K)
    new_points = pointnet_module(new_points)
    return new_points