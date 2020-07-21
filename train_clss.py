import argparse
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from models.pointnet2_cls import pointnet2_cls_ssg, cls_loss
from data.ModelNet40 import ModelNet40


def train_one_epoch(train_loader, model, loss_func, optimizer):
    losses, total_seen, total_correct = [], 0, 0
    for data, labels in train_loader:
        xyz, points = data[:, :3], data[:, 3:]
        pred = model(xyz, points)
        loss = loss_func(pred, labels)

        optimizer.zero_grad()  # Important
        loss.backward()
        optimizer.step()
        pred = torch.max(pred, dim=-1)[1]
        total_correct += torch.sum(pred == labels)
        total_seen += xyz.shape[0]
        losses.append(loss)
    return torch.mean(losses), total_correct / total_seen


def test_one_epoch(model, test_loader, loss_func):
    losses, total_seen, total_correct = [], 0, 0
    for data, labels in test_loader:
        xyz, points = data[:, :3], data[:, 3:]
        pred = model(xyz, points)
        loss = loss_func(pred, labels)

        pred = torch.max(pred, dim=-1)[1]
        total_correct += torch.sum(pred == labels)
        total_seen += xyz.shape[0]
        losses.append(loss)
    return torch.mean(losses), total_correct / total_seen


def train(train_loader, test_loader, model, loss_func, optimizer, scheduler, nepoches, log_interval, log_dir, checkpoint_interval, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    for epoch in nepoches:
        scheduler.step()
        if epoch % checkpoint_interval:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pointnet2_cls_%d.pth" % epoch))
            model.eval()
            lr = model.optimizer.state_dict()['param_groups'][0]['lr']
            loss, acc = test_one_epoch(model, test_loader, loss_func)
            print('='*20, 'Epoch {} / {}, lr: {}'.format(epoch, nepoches, lr), '='*20)
            print('Loss: {}, Acc: {}'.format(loss, acc))

        model.train()
        loss, acc = train_one_epoch(train_loader, model, loss_func, optimizer)
        if epoch % log_interval:
            lr = model.optimizer.state_dict()['param_groups'][0]['lr']
            print('=' * 20, 'Epoch {} / {}, lr: {}'.format(epoch, nepoches, lr), '=' * 20)
            print('Loss: {}, Acc: {}'.format(loss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/data/modelnet40_normal_resampled',
                        help='the root to the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--nclasses', type=int, default=40, help='Number of classes')
    parser.add_argument('--init_lr', type=float, default=0.001, help='Initial learing rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learing rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Adam or momentum')
    parser.add_argument('--decay_step', type=int, default=20000, help='Decay step for lr decay')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay')
    parser.add_argument('--nepoches', type=int, default=251, help='Number of traing epoches')
    parser.add_argument('--log_interval', type=int, default=10, help='Print iterval')
    parser.add_argument('--log_dir', type=str, default='experiments', help='Train/val loss and accuracy logs')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Checkpoint saved interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()

    modelnet40_train = ModelNet40(data_root=args, split='train')
    modelnet40_test = ModelNet40(data_root=args, split='test')
    train_loader = DataLoader(dataset=modelnet40_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=modelnet40_test, batch_size=args.batch_size, shuffle=False)
    model = pointnet2_cls_ssg(args.nclasses)
    loss = cls_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    train(train_loader=train_loader,
          test_loader=test_loader,
          model=model,
          loss_func=loss,
          optimizer=optimizer,
          scheduler=scheduler,
          nepoches=args.nepoches,
          log_interval=args.log_interval,
          log_dir=args.log_dir,
          checkpoint_interval=args.checkpoint_interval,
          checkpoint_dir=args.checkpoint_dir
          )