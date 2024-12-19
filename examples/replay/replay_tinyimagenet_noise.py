################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-10-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Replay strategy.
"""

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
from avalanche.benchmarks import nc_benchmark, SplitTinyImageNet
from avalanche.models import SimpleMLP, SimpleCNN, ResNet18, as_multitask, ResNet34
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin, LRSchedulerPlugin,EarlyStoppingPluginForTrainMetric
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
from examples.fix_seeds import fix_seeds
from run_commands.my_utils.Noise import AddGaussianNoise
from tools import get_args
from tools.SAM.solver.build import build_optimizer


def main(args):
    fix_seeds(args.seed)

    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    n_experiences = 20

    noise_eval_transform = transforms.Compose(
        [
            # AddGaussianNoise(mean=random.uniform(0.5, 1.5), variance=0.5, amplitude=random.uniform(0, 45), p=args.noise_p),
            AddGaussianNoise(mean=0, variance=1, amplitude=255, p=args.noise_p),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
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

    scenario = SplitTinyImageNet(n_experiences=n_experiences,
                                 seed=args.seed,
                                 fixed_class_order=fixed_class_order,
                                 eval_transform=noise_eval_transform,
                                 return_task_id=True if args.scenario == 'task' else False,
                                 class_ids_from_zero_in_each_exp=True if args.scenario == 'task' else False,
                                 train_error=True if args.train_error == 'yes' else False,
                                 )
    model = ResNet34(num_classes=10) if args.scenario == 'task' else ResNet34(num_classes=200) if args.scenario == 'class' else None

    # MODEL CREATION
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
    # optim = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.00001,)
    optim, base_optim = build_optimizer(args, model=model)
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, [30, 70], gamma=0.2)
    )
    early = EarlyStoppingPluginForTrainMetric(patience=10)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        optim,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=100,
        device=device,
        plugins=[ReplayPlugin(mem_size=3000), sched, early],
        evaluator=eval_plugin,
        do_initial=args.do_initial,
        eval_every=0,
        mixup=args.mixup,
        mu=args.mu,
        mix_times=args.mix_times,
        scenario_mode=args.scenario,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    if args.eval_on_random:
        cl_strategy.eval(scenario.test_stream, num_workers=1)
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(scenario.test_stream, num_workers=1))


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
    parser.add_argument('--mixup', type=str, default='no', help="whether or not to use mixup")
    parser.add_argument('--mu', type=float, default=-100)
    parser.add_argument('--mix_times', type=float, default=0.8)
    parser.add_argument('--scenario', type=str, default='task', help="use classIL or taskIL")
    parser.add_argument("--task_mask", type=str, default='no', help="whether use masking in taskIL")
    parser.add_argument("--train_error", type=str, default='no', help="whether only eval on training set")
    parser.add_argument("--noise_p", type=float, default=0.2)
    args = get_args(parser)

    print(args)
    main(args)
