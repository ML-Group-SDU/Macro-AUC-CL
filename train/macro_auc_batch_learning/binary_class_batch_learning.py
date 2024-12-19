import argparse
import json
import os
import shutil
from ctypes import Union
from typing import Any

import torch.multiprocessing
from torch.optim.lr_scheduler import MultiStepLR
import math

from avalanche.benchmarks.datasets import BinarySplitCIFAR10
from avalanche.evaluation.metrics import loss_metrics, forgetting_metrics
from avalanche.evaluation.multilabel_metrics import multilabel_accuracy
from avalanche.logging import InteractiveLogger,CSVLogger,TextLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin, EarlyStoppingPluginForTrainMetric,OneHotPlugin
from examples.utils.utils import fix_seeds
from examples.utils.utils import set_gpu_device
from avalanche.models.simple_cnn import ResNet34,ResNet18,SimpleCNN
from tools.SAM.solver.build import build_optimizer
from tools.myargs import get_args
from avalanche.evaluation.multilabel_metrics.multiLabelLoss import ReweightedLoss
from tools.class_order import *

torch.multiprocessing.set_sharing_strategy('file_system')

def select_binary_class_dataset(args):

    scenario = BinarySplitCIFAR10(
        n_experiences=1,
        seed=args.seed,
        two_class_id=[args.class_id1,args.class_id2],
        train_scalers=args.train_scalers,
        test_scalers=args.test_scalers
    )
    model = SimpleCNN(num_classes=2)

    return scenario,model


def main(args):
    fix_seeds(args.seed)
    device = set_gpu_device(args.cuda)

    scenario, model = select_binary_class_dataset(args)

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
        MultiStepLR(optim, [math.ceil(args.epoch*0.6), math.ceil(args.epoch*0.85)], gamma=0.2)
    )

    # loss function
    if args.crit == "BCE":
        crit = torch.nn.BCEWithLogitsLoss()
    elif args.crit == "ReWeighted":
        crit = ReweightedLoss()
    else:
        raise NotImplementedError

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        optim,
        criterion=crit,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=args.batch_size,
        device=device,
        plugins=[OneHotPlugin(),sched],
        evaluator=eval_plugin,
        do_initial=args.do_initial,
        eval_every=0,
        use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'),
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    if args.eval_on_random:
        cl_strategy.eval(scenario.test_stream,num_workers=2)

    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience, num_workers=2)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(scenario.test_stream,num_workers=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda", type=int, default=1, help="Select zero-indexed cuda device. -1 to use CPU.", )
    parser.add_argument("--do_initial", type=bool, default=False, )
    parser.add_argument("--eval_on_random", type=bool, default=False, )
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    parser.add_argument('--crit',type=str,default='BCE',help="Option between BCE and Reweighted")
    parser.add_argument('--class_id1',type=int, default=0)
    parser.add_argument('--class_id2', type=int, default=1)
    parser.add_argument('--train_scalers',type=str, default='1, 1')
    parser.add_argument('--test_scalers', type=str, default='1, 1')

    args = get_args(parser)
    print(args)
    main(args)
