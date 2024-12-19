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
from avalanche.benchmarks import nc_benchmark, SplitCIFAR10, SplitCIFAR100
from avalanche.models import SimpleMLP, SimpleCNN, ResNet18, as_multitask, ResNet34
from avalanche.training.supervised import Naive
from avalanche.training.plugins import LRSchedulerPlugin, SimilarityReplayPlugin
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


def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda}'
    fix_seeds(args.seed)

    # --- CONFIG
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    print(device)

    n_experiences = 10

    scenario = SplitCIFAR100(n_experiences=n_experiences,
                             seed=args.seed,
                             fixed_class_order= [
                                87,
                                0,
                                52,
                                58,
                                44,
                                91,
                                68,
                                97,
                                51,
                                15,
                                94,
                                92,
                                10,
                                72,
                                49,
                                78,
                                61,
                                14,
                                8,
                                86,
                                84,
                                96,
                                18,
                                24,
                                32,
                                45,
                                88,
                                11,
                                4,
                                67,
                                69,
                                66,
                                77,
                                47,
                                79,
                                93,
                                29,
                                50,
                                57,
                                83,
                                17,
                                81,
                                41,
                                12,
                                37,
                                59,
                                25,
                                20,
                                80,
                                73,
                                1,
                                28,
                                6,
                                46,
                                62,
                                82,
                                53,
                                9,
                                31,
                                75,
                                38,
                                63,
                                33,
                                74,
                                27,
                                22,
                                36,
                                3,
                                16,
                                21,
                                60,
                                19,
                                70,
                                90,
                                89,
                                43,
                                5,
                                42,
                                65,
                                76,
                                40,
                                30,
                                23,
                                85,
                                2,
                                95,
                                56,
                                48,
                                71,
                                64,
                                98,
                                13,
                                99,
                                7,
                                34,
                                55,
                                54,
                                26,
                                35,
                                39,
                            ],
                             return_task_id=True if args.scenario == 'task' else False,
                             class_ids_from_zero_in_each_exp=True if args.scenario == 'task' else False,
                            )
    model = ResNet34(num_classes=10)

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
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001,)

    sched = LRSchedulerPlugin(
        MultiStepLR(optim, [40, 60], gamma=0.2)
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        optim,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=100,
        device=device,
        plugins=[SimilarityReplayPlugin(mem_size=2000), sched],
        evaluator=eval_plugin,
        do_initial=args.do_initial,
        eval_every=0,
        mixup=args.mixup,
        mu=args.mu,
        mix_times=args.mix_times,
        scenario_mode='task',
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
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    parser.add_argument('--mixup', type=str, default='no', help="whether or not to use mixup")
    parser.add_argument('--mu', type=float, default=0.8)
    parser.add_argument('--mix_times', type=float, default=-100)
    parser.add_argument('--scenario', type=str, default='task', help="use classIL or taskIL")
    parser.add_argument("--task_mask", type=str, default='no', help="whether use masking in taskIL")
    args = parser.parse_args()
    print(args)
    main(args)
