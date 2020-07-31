'''
author: charlesq34
addr: https://github.com/charlesq34/pointnet2/blob/master/utils/provider.py

update: zhulf
'''

import numpy as np


def pc_normalize(pc):
    mean = np.mean(pc, axis=0)
    pc -= mean
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)))
    pc /= m
    return pc


def shuffle_points(pc):
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    return pc[idx,:]


def rotate_point_cloud(pc):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_pc = np.dot(pc, rotation_matrix)
    return rotated_pc


def rotate_point_cloud_with_normal(pc_normal):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    pc_normal[:,0:3] = np.dot(pc_normal[:, 0:3], rotation_matrix)
    pc_normal[:,3:6] = np.dot(pc_normal[:, 3:6], rotation_matrix)
    return pc_normal


def rotate_perturbation_point_cloud_with_normal(pc_normal, angle_sigma=0.06, angle_clip=0.18):
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    pc_normal[:,0:3] = np.dot(pc_normal[:, :3], R)
    pc_normal[:,3:6] = np.dot(pc_normal[:, 3:], R)
    return pc_normal


def rotate_point_cloud_by_angle(pc, rotation_angle):
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    pc = np.dot(pc, rotation_matrix)
    return pc


def rotate_point_cloud_by_angle_with_normal(pc_normal, rotation_angle):
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    pc_normal[:, :3] = np.dot(pc_normal[:, :3], rotation_matrix)
    pc_normal[:, 3:6] = np.dot(pc_normal[:, 3:6], rotation_matrix)
    return pc_normal



def rotate_perturbation_point_cloud(pc, angle_sigma=0.06, angle_clip=0.18):
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    pc = np.dot(pc, R)
    return pc


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += pc
    return jittered_data


def shift_point_cloud(pc, shift_range=0.1):
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, (1, C))
    pc += shifts
    return pc


def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    scale = np.random.uniform(scale_low, scale_high, 1)
    pc *= scale
    return pc


def random_point_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc


def augment_pc(pc_normal):
    rotated_pc_normal = rotate_point_cloud_with_normal(pc_normal)
    rotated_pc_normal = rotate_perturbation_point_cloud_with_normal(rotated_pc_normal)
    jittered_pc = random_scale_point_cloud(rotated_pc_normal[:, :3])
    jittered_pc = shift_point_cloud(jittered_pc)
    jittered_pc = jitter_point_cloud(jittered_pc)
    rotated_pc_normal[:, :3] = jittered_pc
    return rotated_pc_normal


if __name__ == '__main__':
    pc = np.random.randn(4, 6)
    print(pc)
    pc = augment_pc(pc)
    print(pc)