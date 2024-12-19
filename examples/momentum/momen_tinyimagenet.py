from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import expanduser
import os
import shutil
import argparse
import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop
import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_benchmark, SplitTinyImageNet
from avalanche.models import SimpleMLP, SimpleCNN, ResNet18, as_multitask, ResNet34
from avalanche.training.supervised import Naive
from avalanche.training.plugins import MomentumPlugin, LRSchedulerPlugin, EarlyStoppingPluginForTrainMetric,ReplayPlugin
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
from examples.fix_seeds import fix_seeds,set_device
from tools import get_args
from tools.SAM.solver.build import build_optimizer
from tools.class_order import tinyimagenet_class_order

def main(args):
    fix_seeds(args.seed)

    # --- CONFIG
    device = set_device(args.cuda)
    n_experiences = 20

    scenario = SplitTinyImageNet(n_experiences=n_experiences,
                                 seed=args.seed,
                                 fixed_class_order=tinyimagenet_class_order,
                                 return_task_id=True if args.scenario == 'task' else False,
                                 # class_ids_from_zero_in_each_exp=True if args.scenario == 'task' or args.scenario == "domain" else False,
                                 class_ids_from_zero_in_each_exp=True if args.scenario == "domain" else False,
                                 train_error=True if args.train_error == 'yes' else False,
                                 )

    if args.scenario == 'task' or args.scenario == "domain":
        model = ResNet34(num_classes=10)
    elif args.scenario == 'class':
        model = ResNet34(num_classes=200)
    else:
        model = None

    # MODEL CREATION
    if args.scenario == 'task':
        model = as_multitask(model, "classifier", masking=True if args.task_mask == "yes" else False)

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

    optim, base_optim = build_optimizer(args, model=model)
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, [30, 70], gamma=0.2)
    )
    # early = EarlyStoppingPluginForTrainMetric(patience=10)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        optim,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=100,
        device=device,
        plugins=[MomentumPlugin(args.alpha),ReplayPlugin(mem_size=3000),sched],
        evaluator=eval_plugin,
        do_initial=args.do_initial,
        eval_every=0,
        scenario_mode=args.scenario,
        use_closure=True if "sam" in args.opt else False
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
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--cuda", type=int, default=1, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--do_initial", type=bool, default=False)
    parser.add_argument("--eval_on_random", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    parser.add_argument('--scenario', type=str, default='task', help="use classIL or taskIL")
    parser.add_argument("--task_mask", type=str, default='yes', help="whether use masking in taskIL")
    parser.add_argument("--train_error", type=str, default='no', help="whether only eval on training set")
    parser.add_argument("--alpha", type=float, default=0.9)


    args = get_args(parser)
    print(args)
    main(args)
