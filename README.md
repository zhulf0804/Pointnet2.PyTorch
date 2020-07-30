## Introduction

An unofficial PyTorch Implementation of [PointNet++: Deep Hierarchical Feature Learning on
Point Sets in a Metric Space]()[NIPS 2017].


## Classification
- **Start**
    - Dataset: [ModelNet40](), download it from [Official Site](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) or [Baidu Disk]().
    - Train
        ```
        python train_clss.py --data_root your_data_root --log_dir your_log_dir

        eg.
        python train_clss.py --data_root /root/modelnet40_normal_resampled --npoints 1024 --log_dir cls_ssg_1024
        ```
    - Evaluate
    
        ```
        python evaluate.py evaluate_cls model data_root checkpoint
        
        eg.
        python evaluate.py evaluate_cls pointnet2_cls_ssg /root/modelnet40_normal_resampled checkpoints/pointnet2_cls_200.pth
        python evaluate.py evaluate_cls pointnet2_cls_msg /root/modelnet40_normal_resampled checkpoints/pointnet2_cls_200.pth
        ```
- **Performance**(the *first row* is the results repored in Paper, the *following rows* are results reported from this repo.)

    | Model | NPoints | Aug | DP | Accuracy(%) |
    | :---: | :---: | :--- : | :---: | :---: |
    | PointNet2(**official**) |  | ✓ | | 91.7 |
    | PointNet2_SSG | | ✗ | | 90.5 |
    | PointNet2_SSG | 4096 | ✗ | ✗ |  |
    | PointNet2_MSG | 1024 | ✗ | ✗ |  |


## Part Segmentation
- Start
- Metrics
- Dataset
- Performance

## Reference