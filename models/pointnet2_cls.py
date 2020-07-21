import torch
import torch.nn as nn
from utils.set_abstraction import pointnet_sa_module, pointnet_sa_module_msg


class pointnet2_cls_ssg(nn.Module):
    def __init__(self, nclasses):
        super(pointnet2_cls_ssg, self).__init__()
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.ReLU(512)
        self.dropout1 = nn.Dropout2d(0.5)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.ReLU(256)
        self.dropout2 = nn.Dropout2d(0.5)
        self.cls = nn.Linear(256, nclasses)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        in_channels = xyz.shape[-1]
        if points is not None:
            in_channels += points.shape[-1]
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
                                                               K=64,
                                                               in_channels=259,
                                                               mlp=[256, 512, 1024],
                                                               group_all=True)
        net = new_points.view(batchsize, -1)
        net = self.dropout1(self.bn1(self.linear1(net)))
        net = self.dropout2(self.bn2(self.linear2(net)))
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
    points = torch.randn(16, 2046, 3)
    label = torch.randint(0, 40, size=(16, ))
    ssg_model = pointnet2_cls_ssg(40)
    net = ssg_model(xyz, points)
    print(net.shape)
    print(label.shape)
    loss = cls_loss()
    loss = loss(net, label)
    print(loss)