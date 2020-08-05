import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.set_abstraction import PointNet_SA_Module, PointNet_SA_Module_MSG
from utils.feature_propagation import PointNet_FP_Module


class pointnet2_seg_ssg(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_seg_ssg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)

        self.pt_fp1 = PointNet_FP_Module(in_channels=1024+256, mlp=[256, 256], bn=True)
        self.pt_fp2 = PointNet_FP_Module(in_channels=256 + 128, mlp=[256, 128], bn=True)
        self.pt_fp3 = PointNet_FP_Module(in_channels=128 + 6, mlp=[128, 128, 128], bn=True)

        self.conv1 = nn.Conv1d(128, 128, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.cls = nn.Conv1d(128, nclasses, 1, stride=1)

    def forward(self, l0_xyz, l0_points):
        l1_xyz, l1_points = self.pt_sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.pt_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.pt_sa3(l2_xyz, l2_points)

        l2_points = self.pt_fp1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.pt_fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.pt_fp3(l0_xyz, l1_xyz, torch.cat([l0_points, l0_xyz], dim=-1), l1_points)

        net = l0_points.permute(0, 2, 1).contiguous()
        net = self.dropout1(F.relu(self.bn1(self.conv1(net))))
        net = self.cls(net)

        return net


class seg_loss(nn.Module):
    def __init__(self):
        super(seg_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, pred, label):
        '''

        :param pred: shape=(B, N, C)
        :param label: shape=(B, N)
        :return:
        '''
        loss = self.loss(pred, label)
        return loss


if __name__ == '__main__':
    in_channels = 6
    n_classes = 50
    l0_xyz = torch.randn(4, 1024, 3)
    l0_points = torch.randn(4, 1024, 3)
    model = pointnet2_seg_ssg(in_channels, n_classes)
    net = model(l0_xyz, l0_points)
    print(net.shape)