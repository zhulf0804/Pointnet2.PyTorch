import torch
import torch.nn as nn
from utils.sampling import fps
from utils.grouping import ball_query
from utils.common import gather_points


class PointNet(nn.Module):
    def __init__(self, in_channels, mlp, bn, pooling):
        super(PointNet, self).__init__()
        self.pooling = pooling
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv{}'.format(i), nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
            if bn:
                self.backbone.add_module('Bn{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, x):
        x = self.backbone(x.float())
        if self.pooling == 'avg':
            x = torch.mean(x, dim=2, keepdim=True)
        else:
            x = torch.max(x, dim=2, keepdim=True)[0]
        return x


def sample_and_group(xyz, points, M, radius, K, use_xyz=True):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_points, shape=(B, M, K, C)
    '''
    new_xyz = gather_points(xyz, fps(xyz, M))
    grouped_inds = ball_query(xyz, new_xyz, radius, K)
    grouped_xyz = gather_points(xyz, grouped_inds)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    B, M, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_inds = torch.arange(0, M).long().view(1, 1, M).repeat(B, 1, 1)
    grouped_xyz = xyz.view(B, 1, M, C)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz.float(), points.float()], dim=2)
        else:
            new_points = points
        new_points = torch.unsqueeze(new_points, dim=1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


def pointnet_sa_module(xyz, points, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
    device = torch.device('cuda')
    pointnet_module = PointNet(in_channels, mlp, bn, pooling).to(device)
    if group_all:
        new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
    else:
        new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz, points, M, radius, K, use_xyz)
    new_points = new_points.permute(0, 3, 2, 1)
    new_points = torch.squeeze(pointnet_module(new_points), 2).permute(0, 2, 1)
    return new_xyz, new_points, grouped_inds


class PointNet_SA_Module(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module, self).__init__()
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1))
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points, grouped_inds


def pointnet_sa_module_msg(xyz, points, M, radius_list, K_list, in_channels, mlp_list, bn=True, pooling='max', use_xyz=True):
    new_xyz = gather_points(xyz, fps(xyz, M))
    new_points_list = []
    for i in range(len(radius_list)):
        mlp, radius, K = mlp_list[i], radius_list[i], K_list[i]
        print(mlp, radius, K, xyz.shape, points.shape)
        new_xyz, new_points, grouped_inds = pointnet_sa_module(xyz, points, M, radius, K, in_channels, mlp, bn, pooling, use_xyz)
        new_points_list.append(new_points)
    new_points_cat = torch.cat(new_points_list, dim=-1)
    return new_xyz, new_points_cat


if __name__ == '__main__':
    import sys
    sys.path.append('/root/code/PointNet2.PyTorch')
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    setup_seed(2)
    xyz = torch.randn(4, 100, 3)
    points = torch.randn(4, 100, 3)

    M, radius, K = 5, 5, 6
    new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points)
    print(new_xyz)
    print(new_points)
    '''
    print('='*20, 'backbone', '='*20)
    M, radius, K, in_channels, mlp = 2, 0.2, 3, 6, [32, 64, 128]
    new_xyz, new_points, grouped_inds = pointnet_sa_module(xyz, points, M, radius, K, in_channels, mlp)
    print('new_xyz: ', new_xyz.shape)
    print('new_points: ', new_points.shape)
    print('grouped_inds: ', grouped_inds.shape)

    print('='*20, 'backbone msg', '='*20)
    M, radius_list, K_list, in_channels, mlp_list = 2, [0.2, 0.4], [3, 4], 6, [[32, 64, 128], [64, 64]]
    new_xyz, new_points_cat = pointnet_sa_module_msg(xyz, points, M, radius_list, K_list, in_channels, mlp_list)
    print('new_xyz: ', new_xyz.shape)
    print(new_points_cat.shape)
    '''