import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/root/code/PointNet2.PyTorch')
from utils.set_abstraction import pointnet_sa_module, pointnet_sa_module_msg, PointNet_SA_Module


class pointnet2_cls_ssg(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_cls_ssg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = nn.Linear(256, nclasses)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        #in_channels = xyz.shape[-1]
        #if points is not None:
        #    in_channels += points.shape[-1]
        '''
        new_xyz, new_points, grouped_inds = pointnet_sa_module(xyz,
                                                               points,
                                                               M=512,
                                                               radius=0.2,
                                                               K=32,
                                                               in_channels=in_channels,
                                                               mlp=[64, 64, 128],
                                                               group_all=False)
        new_xyz, new_points, grouped_inds = pointnet_sa_module(new_xyz,
                                                               new_points,
                                                               M=128,
                                                               radius=0.4,
                                                               K=64,
                                                               in_channels=131,
                                                               mlp=[128, 128, 256],
                                                               group_all=False)
        new_xyz, new_points, grouped_inds = pointnet_sa_module(new_xyz,
                                                               new_points,
                                                               M=None,
                                                               radius=None,
                                                               K=None,
                                                               in_channels=259,
                                                               mlp=[256, 512, 1024],
                                                               group_all=True)
        '''
        new_xyz, new_points, grouped_inds = self.pt_sa1(xyz, points)
        new_xyz, new_points, grouped_inds = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points, grouped_inds = self.pt_sa3(new_xyz, new_points)
        net = new_points.view(batchsize, -1)
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
        net = self.cls(net)
        return net


class pointnet2_cls_msg(nn.Module):
    def __init__(self):
        super(pointnet2_cls_msg, self).__init__()
    def forward(self, x):
        return x


class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, pred, lable):
        '''

        :param pred: shape=(B, nclass)
        :param lable: shape=(B, )
        :return: loss
        '''
        loss = self.loss(pred, lable)
        return loss


if __name__ == '__main__':
    xyz = torch.randn(16, 2048, 3)
    points = torch.randn(16, 2048, 3)
    label = torch.randint(0, 40, size=(16, ))
    ssg_model = pointnet2_cls_ssg(6, 40)
    net = ssg_model(xyz, points)
    print(net.shape)
    print(label.shape)
    loss = cls_loss()
    loss = loss(net, label)
    print(loss)