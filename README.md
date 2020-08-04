## Introduction

An unofficial PyTorch Implementation of [PointNet++: Deep Hierarchical Feature Learning on
Point Sets in a Metric Space]()[NIPS 2017].

### Requirements
- PyTorch, Python3, TensorboardX, tqdm, fire

## Classification
- **Start**
    - Dataset: [ModelNet40](), download it from [Official Site](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) or [Baidu Disk]().
    - Train
        ```
        python train_clss.py --data_root your_data_root --log_dir your_log_dir

        eg.
        python train_clss.py --data_root /root/modelnet40_normal_resampled  --log_dir cls_ssg_1024
        ```
    - Evaluate
    
        ```
        python evaluate.py evaluate_cls model data_root checkpoint npoints
        
        eg.
        python evaluate.py evaluate_cls pointnet2_cls_ssg /root/modelnet40_normal_resampled checkpoints/pointnet2_cls_200.pth 4096
        python evaluate.py evaluate_cls pointnet2_cls_msg /root/modelnet40_normal_resampled checkpoints/pointnet2_cls_200.pth 4096
        ```
- **Performance**(the **first row** is the results reported in Paper, the **following rows** are results reported from this repo.)

    | Model | NPoints | Aug | Accuracy(%) |
    | :---: | :---: | :---: | :---: |
    | PointNet2(**official**) | 5000 | ✓ | 91.7 |
    | PointNet2_SSG | 1024 | ✗ | **91.8** |
    | PointNet2_SSG | 4096 | ✗ | 91.7 |
    | PointNet2_SSG | 4096 | ✓ | 90.5 |
    | PointNet2_MSG | 4096 | ✓ | 91.0 |
    
    | Model | Train_NPoints | DP | Test_NPoints | Accuracy(%) |
    | :---: | :---: | :---: | :---: | :---: |
    | PointNet2_SSG | 1024 | ✗ | 256 | 70.4 |
    | PointNet2_SSG | 1024 | ✓ | 256 |  |
    | PointNet2_SSG | 1024 | ✗ | 1024 | 91.8 |
    | PointNet2_SSG | 1024 | ✓ | 1204 |  |


## Part Segmentation
- **Start**
    - Dataset: [ShapeNet](), download it from [Official Site]() or [Baidu Disk]().
    - Train
        ```
        python train_part_seg.py --data_root your_data_root --log_dir your_log_dir

        eg.
        python train_part_seg.py --data_root /root/shapenetcore_partanno_segmentation_benchmark_v0_normal --log_dir seg_ssg --batch_size 64
        ```
    - Evaluate
    
        ```
        python evaluate.py evaluate_seg model data_root checkpoint npoints
        
        eg.
        python evaluate.py evaluate_seg pointnet2_seg_ssg /root/shapenetcore_partanno_segmentation_benchmark_v0_normal checkpoints/pointnet2_seg_250.pth 2500
        ```
- **Performance**(the **first row** is the results reported in Paper, the **following rows** are results reported from this repo.)


## Reference

- [https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- [https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [https://github.com/sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)