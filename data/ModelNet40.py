import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from data.provider import pc_normalize, rotate_point_cloud_with_normal, rotate_perturbation_point_cloud_with_normal, \
    random_scale_point_cloud, shift_point_cloud, jitter_point_cloud, shuffle_points


class ModelNet40(Dataset):

    def __init__(self, data_root, split, npoints, augment=False, normalize=True):
        assert(split == 'train' or split == 'test')
        self.npoints = npoints
        self.augment = augment
        self.normalize = normalize

        cls2name, name2cls = self.decode_classes(os.path.join(data_root, 'modelnet40_shape_names.txt'))
        train_list_path = os.path.join(data_root, 'modelnet40_train.txt')
        train_files_list = self.read_list_file(train_list_path, name2cls)
        test_list_path = os.path.join(data_root, 'modelnet40_test.txt')
        test_files_list = self.read_list_file(test_list_path, name2cls)
        self.files_list = train_files_list if split == 'train' else test_files_list
        self.caches = {}

    def read_list_file(self, file_path, name2cls):
        base = os.path.dirname(file_path)
        files_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                name = '_'.join(line.strip().split('_')[:-1])
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
        #if self.npoints > 0:
        #    inds = np.random.randint(0, len(xyz_points), size=(self.npoints, ))
        #    xyz_points = xyz_points[inds, :]
        xyz_points = xyz_points[:self.npoints, :]
        if self.normalize:
            xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
        if self.augment:
            xyz_points = self.augment_pc(xyz_points)
        self.caches[index] = [xyz_points, label]
        return xyz_points, label

    def __len__(self):
        return len(self.files_list)


if __name__ == '__main__':
    modelnet40 = ModelNet40(data_root='/root/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled', split='test')
    test_loader = DataLoader(dataset=modelnet40,
                              batch_size=16,
                              shuffle=True)
    for point, label in test_loader:
        print(point.shape)
        print(label.shape)