import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.pointnet2_cls import pointnet2_cls_ssg, pointnet2_cls_msg, cls_loss
from data.CustomDataset import CustomDataset
from utils.common import setup_seed


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


def train(train_loader, test_loader, model, loss_func, optimizer, scheduler, device, ngpus, nepoches, log_interval, log_dir, checkpoint_interval):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)

    for epoch in range(nepoches):
        if epoch % checkpoint_interval == 0:
            print('='*40)
            if ngpus > 1:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, "pointnet2_cls_%d.pth" % epoch))
            else:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pointnet2_cls_%d.pth" % epoch))
            model.eval()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss, total_correct, total_seen, acc = test_one_epoch(test_loader, model, loss_func, device)
            print('Test  Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, total_correct, total_seen, acc))
            writer.add_scalar('test loss', loss, epoch)
            writer.add_scalar('test acc', acc, epoch)
        model.train()
        loss, total_correct, total_seen, acc = train_one_epoch(train_loader, model, loss_func, optimizer, device)
        writer.add_scalar('train loss', loss, epoch)
        writer.add_scalar('train acc', acc, epoch)
        if epoch % log_interval == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('Train Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, total_correct, total_seen, acc))
        scheduler.step()


if __name__ == '__main__':
    Models = {
        'pointnet2_cls_ssg': pointnet2_cls_ssg,
        'pointnet2_cls_msg': pointnet2_cls_msg
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Root to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--npoints', type=int, default=1024, help='Number of the training points')
    parser.add_argument('--nclasses', type=int, required=True, help='Number of classes')
    parser.add_argument('--augment', type=bool, default=False, help='Augment the train data')
    parser.add_argument('--dp', type=bool, default=False, help='Random input dropout during training')
    parser.add_argument('--model', type=str, default='pointnet2_cls_ssg', help='Model name')
    parser.add_argument('--gpus', type=str, default='0', help='Cuda ids')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learing rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Initial learing rate')
    parser.add_argument('--nepoches', type=int, default=251, help='Number of traing epoches')
    parser.add_argument('--step_size', type=int, default=20, help='StepLR step size')
    parser.add_argument('--gamma', type=float, default=0.7, help='StepLR gamma')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--log_interval', type=int, default=10, help='Print iterval')
    parser.add_argument('--log_dir', type=str, default='work_dirs', help='Train/val loss and accuracy logs')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Checkpoint saved interval')
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)

    device_ids = list(map(int, args.gpus.strip().split(','))) if ',' in args.gpus else [int(args.gpus)]
    ngpus = len(device_ids)

    custom_train = CustomDataset(data_root=args.data_root, split='train', npoints=args.npoints, augment=args.augment, dp=args.dp)
    custom_test = CustomDataset(data_root=args.data_root, split='test', npoints=-1)
    train_loader = DataLoader(dataset=custom_train, batch_size=args.batch_size // ngpus, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=custom_test, batch_size=1, shuffle=False, num_workers=1)
    print('Train set: {}'.format(len(custom_train)))
    print('Test set: {}'.format(len(custom_test)))

    Model = Models[args.model]
    model = Model(6, args.nclasses)
    # Mutli-gpus
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    if ngpus > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    loss = cls_loss().to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.7)

    tic = time.time()
    train(train_loader=train_loader,
          test_loader=test_loader,
          model=model,
          loss_func=loss,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device,
          ngpus=ngpus,
          nepoches=args.nepoches,
          log_interval=args.log_interval,
          log_dir=args.log_dir,
          checkpoint_interval=args.checkpoint_interval,
          )
    toc = time.time()
    print('Training completed, {:.2f} minutes'.format((toc - tic) / 60))