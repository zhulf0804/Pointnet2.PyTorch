import json
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.provider import pc_normalize, rotate_point_cloud_with_normal, rotate_perturbation_point_cloud_with_normal, \
    random_scale_point_cloud, shift_point_cloud, jitter_point_cloud, shuffle_points, random_point_dropout


class ShapeNet(Dataset):
    def __init__(self, data_root, split, npoints, augment=False, dp=False, normalize=True):
        assert(split == 'train' or split == 'test' or split == 'val' or split == 'trainval')
        self.npoints = npoints
        self.augment = augment
        self.dp = dp
        self.normalize = normalize
        self.cat = {}
        with open(os.path.join(data_root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f.readlines():
                self.cat[line.strip().split()[0]] = line.strip().split()[1]
        train_json_path = os.path.join(data_root, 'train_test_split', 'shuffled_train_file_list.json')
        val_json_path = os.path.join(data_root, 'train_test_split', 'shuffled_val_file_list.json')
        test_json_path = os.path.join(data_root, 'train_test_split', 'shuffled_test_file_list.json')
        train_lists = self.decode_json(data_root, train_json_path)
        val_lists = self.decode_json(data_root, val_json_path)
        test_lists = self.decode_json(data_root, test_json_path)

        self.file_lists = []
        if split == 'train':
            self.file_lists.extend(train_lists)
        elif split == 'val':
            self.file_lists.extend(val_lists)
        elif split == 'trainval':
            self.file_lists.extend(train_lists)
            self.file_lists.extend(val_lists)
        elif split == 'test':
            self.file_lists.extend(test_lists)

        self.seg_classes = {'Earphone': [16, 17, 18],
                            'Motorbike': [30, 31, 32, 33, 34, 35],
                            'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11],
                            'Laptop': [28, 29], 'Cap': [6, 7],
                            'Skateboard': [44, 45, 46], 'Mug': [36, 37],
                            'Guitar': [19, 20, 21], 'Bag': [4, 5],
                            'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                            'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.caches = {}

    def decode_json(self, data_root, path):
        with open(path, 'r') as f:
            l = json.load(f)
        l = [os.path.join(data_root, item.split('/')[1], item.split('/')[2] + '.txt') for item in l]
        return l

    def augment_pc(self, pc_normal):
        rotated_pc_normal = rotate_point_cloud_with_normal(pc_normal)
        rotated_pc_normal = rotate_perturbation_point_cloud_with_normal(rotated_pc_normal)
        jittered_pc = random_scale_point_cloud(rotated_pc_normal[:, :3])
        jittered_pc = shift_point_cloud(jittered_pc)
        jittered_pc = jitter_point_cloud(jittered_pc)
        rotated_pc_normal[:, :3] = jittered_pc
        return rotated_pc_normal

    def __getitem__(self, index):
        if index in self.caches:
            xyz_points, labels = self.caches[index]
        else:
            pc = np.loadtxt(self.file_lists[index]).astype(np.float32)
            xyz_points = pc[:, :6]
            labels = pc[:, -1].astype(np.int32)

            if self.normalize:
                xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
            if self.augment:
                xyz_points = self.augment_pc(xyz_points)
            if self.dp:
                xyz_points = random_point_dropout(xyz_points)
            self.caches[index] = xyz_points, labels

        # resample
        choice = np.random.choice(len(xyz_points), self.npoints, replace=True)
        xyz_points = xyz_points[choice, :]
        labels = labels[choice]
        return xyz_points, labels

    def __len__(self):
        return len(self.file_lists)


if __name__ == '__main__':
    shapenet = ShapeNet(data_root='/root/shapenetcore_partanno_segmentation_benchmark_v0_normal', split='test', npoints=2500)
    print(shapenet.__len__())
    print(shapenet.__getitem__(0))