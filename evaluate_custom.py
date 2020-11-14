import fire
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.pointnet2_cls import pointnet2_cls_ssg, pointnet2_cls_msg
from data.CustomDataset import CustomDataset
from utils.common import setup_seed


def evaluate_cls(model_id, data_root, checkpoint, nclasses, npoints=-1, dims=6):
    setup_seed(222)
    print('Loading..')
    Models = {
        'pointnet2_cls_ssg': pointnet2_cls_ssg,
        'pointnet2_cls_msg': pointnet2_cls_msg
    }
    Model = Models[model_id]
    custom_test = CustomDataset(data_root=data_root, split='test', npoints=npoints)
    test_loader = DataLoader(dataset=custom_test,
                             batch_size=1, shuffle=False,
                             num_workers=1)
    device = torch.device('cuda')
    model = Model(dims, nclasses)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('Loading {} completed'.format(checkpoint))
    print("Dataset: {}, Evaluating..".format(len(custom_test)))
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


if __name__ == '__main__':
    fire.Fire()