## Introduction

An unofficial PyTorch Implementation of [PointNet++: Deep Hierarchical Feature Learning on
Point Sets in a Metric Space]()[NIPS 2017].

### Requirements
- PyTorch, Python3, TensorboardX, tqdm, fire

## Classification
- **Start**
    - Dataset: [ModelNet40](https://modelnet.cs.princeton.edu/), download it from [Official Site](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) or [Baidu Disk](https://pan.baidu.com/s/1E0DqMLebg89IzrXlB-YVDA) with **hi1i**.
    - Train
        ```
        python train_clss.py --data_root your_data_root --log_dir your_log_dir

        eg.
        python train_clss.py --data_root /root/modelnet40_normal_resampled --log_dir cls_ssg_1024
        ```
    - Evaluate
    
        ```
        python evaluate.py evaluate_cls model data_root checkpoint npoints
        
        eg.
        python evaluate.py evaluate_cls pointnet2_cls_ssg  /root/modelnet40_normal_resampled \
        checkpoints/pointnet2_cls_250.pth 1024
        
        python evaluate.py evaluate_cls pointnet2_cls_msg root/modelnet40_normal_resampled \
        checkpoints/pointnet2_cls_250.pth 1024
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
    | PointNet2_SSG | 1024 | ✗ | 256 | 67.9 |
    | PointNet2_SSG | 1024 | ✓ | **256** | **90.8** |
    | PointNet2_SSG | 1024 | ✗ | 1024 | 91.8 |
    | PointNet2_SSG | 1024 | ✓ | 1204 | **91.9** |


## Part Segmentation
- **Start**
    - Dataset: [ShapeNet part](https://shapenet.cs.stanford.edu/iccv17/#dataset), download it from [Official Site](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) or [Baidu Disk](https://pan.baidu.com/s/18YoYMam3vVVqE5i6BXU5kw) with **3e5z**.
    - Train
        ```
        python train_part_seg.py --data_root your_data_root --log_dir your_log_dir

        eg.
        python train_part_seg.py --data_root /root/shapenetcore_partanno_segmentation_benchmark_v0_normal \
        --log_dir seg_ssg --batch_size 64
        ```
    - Evaluate
    
        ```
        python evaluate.py evaluate_seg data_root checkpoint
        
        eg.
        python evaluate.py evaluate_seg /root/shapenetcore_partanno_segmentation_benchmark_v0_normal \
        seg_ssg/checkpoints/pointnet2_cls_250.pth
        ```
- **Metrics**: [Average IoU](https://shapenet.cs.stanford.edu/iccv17/#evaluation)
    
    | Model | Metrics | mean | aero | bag | cap | car | chair | ear phone | guitar | knife | lamp | laptop | motor | mug | pistol | rocket | skate board | table |
    | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
    | PointNet2(**official**) | IoU | 85.1 | 82.4 | 79.0 | 87.7 | 77.3 | 90.8 | 71.8 | 91.0 | 85.9 | 83.7 | 95.3 | 71.6 | 94.1 | 81.3 | 58.7 | 76.4 | 82.6 |
    | PointNet2_SSG | IoU | 84.1 | 82.3 | 75.0 | 80.1 | 77.8 | 90.2 | 73.7 | 90.7 | 84.1 | 82.9 | 95.0 | 69.3 | 93.3 | 80.3 | 55.6 | 76.3 | 80.7 |
    | PointNet2_SSG | Accuracy | 93.2 | 89.9 | 89.0 | 85.5 | 91.8 | 94.4 | 93.5 | 96.1 | 91.1 | 89.2 | 96.9 | 87.4 | 96.4 | 93.7 | 77.2 | 95.9 | 94.8 |
  

## Reference

- [https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- [https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [https://github.com/sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)