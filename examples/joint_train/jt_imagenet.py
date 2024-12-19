from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import os
import shutil

from torch.optim.lr_scheduler import MultiStepLR

from avalanche.benchmarks.classic import SplitTinyImageNet
from avalanche.models import SimpleMLP, as_multitask
from avalanche.training.supervised import JointTraining
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)
from avalanche.models import ResNet34
from avalanche.logging import InteractiveLogger, TextLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin

from examples.fix_seeds import fix_seeds, set_device
from tools.class_order import tinyimagenet_class_order


def main(args):
    # Config
    fix_seeds(2222)
    device = set_device(args.cuda)

    # model
    model = ResNet34(num_classes=10) if (args.scenario == 'task' or args.scenario == 'domain') else ResNet34(num_classes=200) if args.scenario == 'class' else None

    if args.scenario == 'task':
        model = as_multitask(model, "classifier", masking=False)

    # CL Benchmark Creation
    tinyi = SplitTinyImageNet(n_experiences=20,
                              seed=args.seed,
                              return_task_id=True if args.scenario == 'task' else False,
                              class_ids_from_zero_in_each_exp=True if args.scenario == 'domain' else False,
                              fixed_class_order=tinyimagenet_class_order,
                              )
    train_stream = tinyi.train_stream
    test_stream = tinyi.test_stream

    # Prepare for training & testing
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001, )
    criterion = CrossEntropyLoss()

    if os.path.isdir(args.csvlog_dir):
        shutil.rmtree(args.csvlog_dir)
    interactive_logger = InteractiveLogger()
    csv_logger = CSVLogger(log_folder=args.csvlog_dir)
    txt_logger = TextLogger(open(args.csvlog_dir + "/" + args.txtlog_name, 'a+'))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, txt_logger],
    )
    sched = LRSchedulerPlugin(
        MultiStepLR(optimizer, [int(args.epoch * 0.6), int(args.epoch * 0.85)], gamma=0.2)
    )
    # Joint training strategy
    joint_train = JointTraining(
        model,
        optimizer,
        criterion,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=128,
        device=device,
        plugins=[eval_plugin, sched],
    )

    # train and test loop
    results = []
    print("Starting training.")
    # Differently from other avalanche strategies, you NEED to call train
    # on the entire stream.
    joint_train.train(train_stream)
    results.append(joint_train.eval(test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument('--scenario', type=str, default='class', help="use classIL or taskIL")
    args = parser.parse_args()

    main(args)
