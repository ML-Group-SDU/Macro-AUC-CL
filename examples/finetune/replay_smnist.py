
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import expanduser
import os
import shutil
import argparse
import torch
from torch.nn import CrossEntropyLoss,BCELoss
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop
import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_benchmark, SplitMNIST
from avalanche.models import SimpleMLP, SimpleCNN, ResNet18, as_multitask, ResNet34
from avalanche.training.supervised import Naive
from avalanche.training.plugins import LRSchedulerPlugin
from torch.optim.lr_scheduler import MultiStepLR
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    forward_transfer_metrics,
    bwt_metrics,
    confusion_matrix_metrics,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger, CSVLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
import numpy as np
import random
from examples.fix_seeds import fix_seeds, set_device


def main(args):
    fix_seeds(args.seed)
    device = set_device(args.cuda)

    n_experiences = 5

    scenario = SplitMNIST(n_experiences=n_experiences,
                            seed=args.seed,
                            fixed_class_order= [1,2,4,5,6,8,7,3,0,9],
                            return_task_id=True if args.scenario == 'task' else False,
                            class_ids_from_zero_in_each_exp=True if args.scenario == 'domain' else False,
                            )

    # MODEL CREATION
    if args.scenario == 'task' or args.scenario == "domain":
        model = SimpleMLP(num_classes=2,hidden_layers=2)
    elif args.scenario == "class":
        model = SimpleMLP(num_classes=10,hidden_layers=2)
    else:
        model = None

    if args.scenario == 'task':
        model = as_multitask(model, "classifier", masking=True if args.task_mask == 'yes' else False)

    if os.path.isdir(args.csvlog_dir):
        shutil.rmtree(args.csvlog_dir)

    #  choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    #  t_logger = TensorboardLogger()
    csv_logger = CSVLogger(log_folder=args.csvlog_dir)
    txt_logger = TextLogger(open(args.csvlog_dir + "/" + args.txtlog_name, 'a+'))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # forward_transfer_metrics(experience=True, stream=True),
        loggers=[interactive_logger, csv_logger, txt_logger],
    )
    # optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001,)


    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        optim,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=100,
        device=device,
        plugins=[],
        evaluator=eval_plugin,
        do_initial=args.do_initial,
        eval_every=0,
        scenario_mode=args.scenario,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    if args.eval_on_random:
        cl_strategy.eval(scenario.test_stream)
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(scenario.test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.", )
    parser.add_argument("--do_initial", type=bool, default=False, )
    parser.add_argument("--eval_on_random", type=bool, default=True, )
    parser.add_argument("--seed", type=int, default=1025, )
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    parser.add_argument('--scenario', type=str, default='task', help="use classIL or taskIL")
    parser.add_argument("--task_mask", type=str, default='no', help="whether use masking in taskIL")
    parser.add_argument("--train_error", type=str, default='no', help="whether only eval on training set")
    args = parser.parse_args()
    print(args)
    main(args)
