from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import expanduser
import os
import shutil
import argparse
import torch
from torch.nn import CrossEntropyLoss, BCELoss
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from avalanche.benchmarks.classic.ctiny_imagenet import _get_tiny_imagenet_dataset
from avalanche.benchmarks.datasets import TinyImagenet
from avalanche.models import ResNet34
from avalanche.training.plugins import  LRSchedulerPlugin, EarlyStoppingPluginForTrainMetric
from torch.optim.lr_scheduler import MultiStepLR
from examples.fix_seeds import fix_seeds
from tools import get_args
from tools.SAM.solver.build import build_optimizer
from torchvision.transforms import transforms


def main(args):
    fix_seeds(args.seed)

    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    fixed_class_order = [195, 144, 36, 11, 170, 153, 44, 72, 15, 168, 126, 151, 70, 130,
                         137, 65, 22, 92, 1, 42, 9, 8, 19, 115, 77, 123, 118, 182,
                         108, 160, 178, 93, 80, 18, 48, 2, 99, 122, 24, 152, 63, 179,
                         166, 75, 69, 154, 159, 119, 172, 14, 162, 31, 40, 135, 184, 94,
                         158, 25, 89, 68, 147, 155, 175, 98, 186, 60, 134, 67, 97, 197,
                         86, 56, 109, 34, 106, 96, 164, 28, 58, 143, 128, 90, 145, 102,
                         176, 38, 189, 190, 121, 57, 51, 55, 163, 138, 169, 188, 177, 104,
                         148, 61, 136, 114, 174, 111, 79, 7, 13, 87, 192, 101, 54, 180,
                         113, 161, 194, 193, 83, 142, 131, 129, 64, 171, 105, 146, 73, 112,
                         29, 37, 12, 82, 78, 150, 199, 81, 95, 45, 185, 141, 157, 39,
                         10, 125, 43, 107, 198, 76, 26, 120, 117, 173, 187, 140, 110, 181,
                         4, 33, 183, 62, 49, 17, 91, 50, 127, 6, 196, 132, 47, 53,
                         30, 156, 27, 5, 3, 165, 71, 116, 133, 88, 167, 0, 16, 23,
                         191, 35, 59, 32, 85, 46, 74, 21, 103, 52, 100, 149, 20, 124,
                         139, 66, 41, 84]
    _default_train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    _default_eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    train_set = TinyImagenet(root=None, train=True,transform=_default_train_transform)

    test_set = TinyImagenet(root=None, train=False,transform=_default_eval_transform)

    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  )
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args.batch_size,
                             shuffle=True)

    model = ResNet34(num_classes=200).to(device)


    if os.path.isdir(args.csvlog_dir):
        shutil.rmtree(args.csvlog_dir)

    optim, base_optim = build_optimizer(args, model=model)
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, [30, 70], gamma=0.2)
    )
    early = EarlyStoppingPluginForTrainMetric(patience=10)
    Loss_fuc = BCELoss()
    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for i in range(args.epoch):
        loss = 0
        for x, y in train_dataloader:
            x.to(device)
            y.to(device)
            y_hat = model(x)
            loss += Loss_fuc(y, y_hat)
            loss.back_ward()
            optim.step()
        print("Loss:",loss)

        for x, y in test_loader:
            x.to(device)
            y.to(device)
            y_hat = model(x)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda", type=int, default=1, help="Select zero-indexed cuda device. -1 to use CPU.", )
    parser.add_argument("--do_initial", type=bool, default=False, )
    parser.add_argument("--eval_on_random", type=bool, default=True, )
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    parser.add_argument('--mixup', type=str, default='first_exp_mixup', help="whether or not to use mixup")
    parser.add_argument('--mu', type=float, default=-100)
    parser.add_argument('--mix_times', type=float, default=0.8)
    parser.add_argument('--scenario', type=str, default='task', help="use classIL or taskIL")
    parser.add_argument("--task_mask", type=str, default='no', help="whether use masking in taskIL")
    parser.add_argument("--train_error", type=str, default='no', help="whether only eval on training set")
    parser.add_argument("--noise_p", type=float, default=0.2)
    args = get_args(parser)

    print(args)
    main(args)
