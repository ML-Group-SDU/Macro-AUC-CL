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
from avalanche.benchmarks import nc_benchmark, SplitMNIST
from avalanche.models import SimpleMLP, SimpleCNN, ResNet18, as_multitask, ResNet34
from avalanche.training.supervised import CoPE
from avalanche.training.plugins import ReplayPlugin, LRSchedulerPlugin
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
    n_experiences = 5
    multi_task_mode = args.multi_task


    scenario = SplitMNIST(n_experiences=n_experiences,
                            seed=args.seed,
                            fixed_class_order= [1,2,4,5,6,8,7,3,0,9],
                            return_task_id=multi_task_mode,
                            class_ids_from_zero_in_each_exp=multi_task_mode,
                            )
    model = SimpleMLP(
        num_classes=10,
        hidden_layers=2
    )

    # MODEL CREATION

    if multi_task_mode:
        model = as_multitask(model, "classifier", masking=not args.no_masking)

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
    cl_strategy = CoPE(
        model,
        optim,
        mem_size=1000,
        n_classes=10,
        p_size=512,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
        do_initial=args.do_initial,
        eval_every=0,
        mixup=args.mixup,
        mu=args.mu,
        mix_multi=args.mix_multi,
        scenario_mode='class',
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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    parser.add_argument('--mixup', action="store_true")
    parser.add_argument('--mu', type=float, default=-100)
    parser.add_argument('--mix_multi', type=float, default=1)
    parser.add_argument('--multi_task', action='store_true')
    parser.add_argument("--no_masking", action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
