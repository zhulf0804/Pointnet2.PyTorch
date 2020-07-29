import argparse
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from models.pointnet2_cls import pointnet2_cls_ssg, cls_loss
from data.ModelNet40 import ModelNet40

torch.backends.cudnn.enabled = False  # why need ?
'''
Traceback (most recent call last):
  File "train_clss.py", line 108, in <module>
    checkpoint_dir=args.checkpoint_dir
  File "train_clss.py", line 57, in train
    loss, total_correct, total_seen, acc = train_one_epoch(train_loader, model, loss_func, optimizer, device)
  File "train_clss.py", line 19, in train_one_epoch
    loss.backward()
  File "/root/miniconda3/lib/python3.7/site-packages/torch/tensor.py", line 195, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/root/miniconda3/lib/python3.7/site-packages/torch/autograd/__init__.py", line 99, in backward
    allow_unreachable=True)  # allow_unreachable flag
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
'''

def train_one_epoch(train_loader, model, loss_func, optimizer, device):
    losses, total_seen, total_correct = [], 0, 0
    for data, labels in train_loader:
        optimizer.zero_grad()  # Important
        labels = labels.to(device)
        xyz, points = data[:, :, :3], data[:, :, 3:]
        pred = model(xyz.to(device), points.to(device))
        loss = loss_func(pred, labels)

        loss.backward()
        optimizer.step()
        pred = torch.max(pred, dim=-1)[1]
        total_correct += torch.sum(pred == labels)
        total_seen += xyz.shape[0]
        losses.append(loss.item())
    return np.mean(losses), total_correct, total_seen, total_correct / float(total_seen)


def test_one_epoch(test_loader, model, loss_func, device):
    losses, total_seen, total_correct = [], 0, 0
    for data, labels in test_loader:
        labels = labels.to(device)
        xyz, points = data[:, :, :3], data[:, :, 3:]
        with torch.no_grad():
            pred = model(xyz.to(device), points.to(device))
            loss = loss_func(pred, labels)

            pred = torch.max(pred, dim=-1)[1]
            total_correct += torch.sum(pred == labels)
            total_seen += xyz.shape[0]
            losses.append(loss.item())
    return np.mean(losses), total_correct, total_seen, total_correct / float(total_seen)


def train(train_loader, test_loader, model, loss_func, optimizer, scheduler, device, nepoches, log_interval, log_dir, checkpoint_interval, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    for epoch in range(nepoches):
        if epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pointnet2_cls_%d.pth" % epoch))
            model.eval()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('=' * 20, 'Epoch {} / {}, lr: {}'.format(epoch, nepoches, lr), '=' * 20)
            loss, total_correct, total_seen, acc = test_one_epoch(test_loader, model, loss_func, device)
            print('Loss: {}, Corr: {}, Total: {}, Acc: {}'.format(loss, total_correct, total_seen, acc))

        model.train()
        loss, total_correct, total_seen, acc = train_one_epoch(train_loader, model, loss_func, optimizer, device)
        if epoch % log_interval == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('=' * 20, 'Epoch {} / {}, lr: {}'.format(epoch, nepoches, lr), '=' * 20)
            print('Loss: {}, Corr: {}, Total: {}, Acc: {}'.format(loss, total_correct, total_seen, acc))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/modelnet40_normal_resampled',
                        help='the root to the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--nclasses', type=int, default=40, help='Number of classes')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learing rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learing rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Initial learing rate')
    parser.add_argument('--nepoches', type=int, default=251, help='Number of traing epoches')
    parser.add_argument('--log_interval', type=int, default=1, help='Print iterval')
    parser.add_argument('--log_dir', type=str, default='experiments', help='Train/val loss and accuracy logs')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Checkpoint saved interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()

    modelnet40_train = ModelNet40(data_root=args.data_root, split='train')
    modelnet40_test = ModelNet40(data_root=args.data_root, split='test')
    train_loader = DataLoader(dataset=modelnet40_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=modelnet40_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    device = torch.device('cuda')
    model = pointnet2_cls_ssg(6, args.nclasses).to(device)
    loss = cls_loss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum)
    '''
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.init_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    '''
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    train(train_loader=train_loader,
          test_loader=test_loader,
          model=model,
          loss_func=loss,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device,
          nepoches=args.nepoches,
          log_interval=args.log_interval,
          log_dir=args.log_dir,
          checkpoint_interval=args.checkpoint_interval,
          checkpoint_dir=args.checkpoint_dir
          )
