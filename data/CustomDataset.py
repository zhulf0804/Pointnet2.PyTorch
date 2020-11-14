import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from data.provider import pc_normalize, rotate_point_cloud_with_normal, rotate_perturbation_point_cloud_with_normal, \
    random_scale_point_cloud, shift_point_cloud, jitter_point_cloud, shuffle_points, random_point_dropout


class CustomDataset(Dataset):
    def __init__(self, data_root, split, npoints, augment=False, dp=False, normalize=True):
        assert(split == 'train' or split == 'test')
        self.npoints = npoints
        self.augment = augment
        self.dp = dp
        self.normalize = normalize

        cls2name, name2cls = self.decode_classes(os.path.join(data_root, 'shape_names.txt'))
        train_list_path = os.path.join(data_root, 'train.txt')
        train_files_list = self.read_list_file(train_list_path, name2cls)
        test_list_path = os.path.join(data_root, 'test.txt')
        test_files_list = self.read_list_file(test_list_path, name2cls)
        self.files_list = train_files_list if split == 'train' else test_files_list
        self.caches = {}

    def read_list_file(self, file_path, name2cls):
        base = os.path.dirname(file_path)
        files_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                name = line.strip().split('_')[0]
                cur = os.path.join(base, name, '{}.txt'.format(line.strip()))
                files_list.append([cur, name2cls[name]])
        return files_list

    def decode_classes(self, file_path):
        cls2name, name2cls = {}, {}
        with open(file_path, 'r') as f:
            for i, name in enumerate(f.readlines()):
                cls2name[i] = name.strip()
                name2cls[name.strip()] = i
        return cls2name, name2cls

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
            return self.caches[index]
        file, label = self.files_list[index]
        xyz_points = np.loadtxt(file, delimiter=',')
        if self.npoints > 0:
            inds = np.random.randint(0, len(xyz_points), size=(self.npoints, ))
        else:
            inds = np.arange(len(xyz_points))
            np.random.shuffle(inds)
        xyz_points = xyz_points[inds, :]
        if self.normalize:
            xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
        if self.augment:
            xyz_points = self.augment_pc(xyz_points)
        if self.dp:
            xyz_points = random_point_dropout(xyz_points)
        self.caches[index] = [xyz_points, label]
        return xyz_points, label

    def __len__(self):
        return len(self.files_list)