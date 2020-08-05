import fire
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.pointnet2_cls import pointnet2_cls_ssg, pointnet2_cls_msg
from models.pointnet2_seg import pointnet2_seg_ssg
from data.ModelNet40 import ModelNet40
from data.ShapeNet import ShapeNet
from utils.IoU import cal_accuracy_iou


def evaluate_cls(model_id, data_root, checkpoint, npoints, dims=6, nclasses=40):
    print('Loading..')
    Models = {
        'pointnet2_cls_ssg': pointnet2_cls_ssg,
        'pointnet2_cls_msg': pointnet2_cls_msg
    }
    Model = Models[model_id]
    modelnet40_test = ModelNet40(data_root=data_root, split='test', npoints=npoints)
    test_loader = DataLoader(dataset=modelnet40_test,
                             batch_size=64, shuffle=False,
                             num_workers=1)
    device = torch.device('cuda')
    model = Model(dims, nclasses)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('Loading {} completed'.format(checkpoint))
    print("Dataset: {}, Evaluating..".format(len(modelnet40_test)))
    total_correct, total_seen = 0, 0
    for data, labels in tqdm(test_loader):
        labels = labels.to(device)
        xyz, points = data[:, :, :3], data[:, :, 3:]
        with torch.no_grad():
            pred = model(xyz.to(device), points.to(device))
            pred = torch.max(pred, dim=-1)[1]
            total_correct += torch.sum(pred == labels)
            total_seen += xyz.shape[0]
    print("Evaluating completed!")
    print('Corr: {}, Seen: {}, Acc: {:.4f}'.format(total_correct, total_seen, total_correct / float(total_seen)))


def evaluate_seg(data_root, checkpoint, npoints=2048, dims=6, nclasses=50):
    print('Loading..')
    shapenet_test = ShapeNet(data_root=data_root, split='test', npoints=npoints)
    test_loader = DataLoader(dataset=shapenet_test, batch_size=64, shuffle=False, num_workers=4)
    device = torch.device('cuda')
    model = pointnet2_seg_ssg(dims, nclasses)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('Loading {} completed'.format(checkpoint))
    print("Dataset: {}, Evaluating..".format(len(shapenet_test)))
    preds, labels = [], []
    for data, label in tqdm(test_loader):
        labels.append(label)
        xyz, points = data[:, :, :3], data[:, :, 3:]
        with torch.no_grad():
            pred = model(xyz.to(device), points.to(device))
            pred = torch.max(pred, dim=1)[1].cpu().detach().numpy()
            preds.append(pred)
    iou, acc = cal_accuracy_iou(np.concatenate(preds, axis=0), np.concatenate(labels, axis=0), shapenet_test.seg_classes)
    print("Weighed Acc: {:.4f}".format(acc))
    print("Weighed Average IoU: {:.4f}".format(iou))
    print('='*40)
    print("Evaluating completed !")


if __name__ == '__main__':
    fire.Fire()