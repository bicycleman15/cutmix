# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import logging
import os
import shutil
import time
from cv2 import transform
from numpy import mod
import pandas as pd
# import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.model_selection._split import StratifiedShuffleSplit
from theconf.argument_parser import ConfigArgumentParser
from torch.utils.data.dataset import Subset
from torchvision.transforms.transforms import RandomRotation, RandomVerticalFlip
from tqdm import tqdm

from network import resnet as RN
import network.pyramidnet as PYRM
from network.wideresnet import WideResNet as WRN
import utils
import warnings

from cutmix.cutmix import CutMix

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = ConfigArgumentParser(conflict_handler='resolve')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--expname', default='TEST', type=str, help='name of experiment')
parser.add_argument('--cifarpath', default='../data', type=str)
parser.add_argument('--imagenetpath', default="/home/jatin/Imagenet/ILSVRC/Data/CLS-LOC", type=str)
parser.add_argument('--autoaug', default='', type=str)
parser.add_argument('--cv', default=-1, type=int)
parser.add_argument('--only-eval', action='store_true')
parser.add_argument('--checkpoint', default='', type=str)

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

from time import localtime, strftime
import errno
from util.logger import Logger

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def main():
    global args
    args = parser.parse_args()

    # create model save path
    current_time = strftime("%d-%b", localtime())
    model_save_folder = f"checkpoints/{current_time}_{args.experiment_name}"
    mkdir_p(model_save_folder)
    logging.basicConfig(level=logging.INFO, 
                        # format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(model_save_folder, "train.log")),
                            logging.StreamHandler()
                        ])
    logging.info(f"Creating model save path: {model_save_folder}")

    log_path = os.path.join(model_save_folder, "train_metrics.txt")
    logger = Logger(log_path, resume=os.path.isfile(log_path))
    logger.set_names(["epoch", "lr", "train_loss", "top1_train", "test_loss", "top1", "top5"])

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )

        transform_pre_cnc = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
        ])

        transform_post_cnc = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            ds_train = datasets.CIFAR100(args.cifarpath, train=True, download=True, transform=transform_pre_cnc)
            ds_test = datasets.CIFAR100(args.cifarpath, train=False, transform=transform_test)

            train_loader = torch.utils.data.DataLoader(
                CutMix(ds_train, 100, beta=args.cutmix_beta, prob=args.cutmix_prob, num_mix=args.cutmix_num, transform=(transform_test, transform_post_cnc)),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(ds_test,
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            ds_train = datasets.CIFAR10(args.cifarpath, train=True, download=True, transform=transform_pre_cnc)

            train_loader = torch.utils.data.DataLoader(
                CutMix(ds_train, 10,
                beta=args.cutmix_beta, prob=args.cutmix_prob, num_mix=args.cutmix_num, transform=(transform_test, transform_post_cnc)),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.cifarpath, train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenetpath, 'train')
        valdir = os.path.join(args.imagenetpath, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        transform_pre_cnc = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])

        transform_post_cnc = transforms.Compose([
            transforms.ToTensor(),
            jittering,
            lighting,
            normalize
        ])

        train_dataset = datasets.ImageFolder(traindir, transform=transform_pre_cnc)
        train_dataset = CutMix(train_dataset, 1000, beta=args.cutmix_beta, prob=args.cutmix_prob, num_mix=args.cutmix_num, transform=transform_post_cnc)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        numberofclass = 1000
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    logging.info("Creating model {}-{}".format(args.net_type, args.depth))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass+1, True)
    else:
        raise ValueError('unknown network architecture: {}'.format(args.net_type))

    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=1e-4, nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule_steps)

    cudnn.benchmark = True

    best_acc1 = 0.
    best_acc5 = 0.

    for epoch in range(0, args.epochs):

        # train for one epoch
        top1_train, _, train_loss = run_epoch(train_loader, model, criterion, optimizer, epoch, 'train')

        # evaluate on test set
        top1, top5, val_loss = run_epoch(val_loader, model, criterion, None, epoch, 'test')

        # remember best prec@1 and save checkpoint
        is_best = top1 > best_acc1
        best_acc1 = max(top1, best_acc1)
        if is_best:
            best_acc5 = top5
            logging.info('New best (top-1 and 5) obtained: {:.5f} and {:.5f}'.format(best_acc1, best_acc5))

        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=model_save_folder, filename="checkpoint.pth")

        scheduler.step()

        logger.append([epoch, get_learning_rate(optimizer), train_loss, top1_train, val_loss, top1, top5])
        logging.info("epoch {} end stats: train_loss : {:.5f} | test_loss : {:.5f} | top1 : {:.5f} | top5 : {:.5f}".format(epoch, train_loss, val_loss, top1, top5))

    logging.info('Best(top-1 and 5): {:.5f} and {:.5f}'.format(best_acc1, best_acc5))
    logger.close()


def run_epoch(loader, model, criterion, optimizer, epoch, tag):
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_corr = AverageMeter()

    if tag == "train":
        model.train()
    else:
        model.eval()

    end = time.time()

    if optimizer:
        current_lr = get_learning_rate(optimizer)
    else: 
        current_lr = 0.

    logging.info("{} : epoch {}/{} with lr : {:.5f}".format(tag, epoch, args.epochs, current_lr))
    loader = tqdm(loader)

    for i, data in enumerate(loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        if tag == "train":
            input, target, corrupted_img, corr_target = data
            input, target, corrupted_img, corr_target = input.cuda(), target.cuda(), corrupted_img.cuda(), corr_target.cuda()
        else:
            input, target = data
            input, target = input.cuda(), target.cuda()

        if tag == "train":
            output = model(input)
            output_corr = model(corrupted_img)
            loss = criterion(output, target) + criterion(output_corr, corr_target)

        else:
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        if len(target.shape) == 1:
            if tag == "train":
                err1 = accuracy(output.data, target, topk=(1,))
                top1.update(err1.item(), input.size(0))
                err1 = accuracy(output_corr.data, corr_target, topk=(1,))
                top1_corr.update(err1.item(), input.size(0))
            else:
                err1, err5 = accuracy(output.data, target, topk=(1, 5))
                top1.update(err1.item(), input.size(0))
                top5.update(err5.item(), input.size(0))

        if optimizer:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        if tag == "train":
            loader.set_postfix_str("loss: {:.5f} | train_top1: {:.5f} | train_top1_corr : {:.5f}".format(losses.avg, top1.avg, top1_corr.avg))
        else:
            loader.set_postfix_str("loss: {:.5f} | err1: {:.5f} | err5: {:.5f}".format(losses.avg, top1.avg, top5.avg))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f"model_best.pth"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
