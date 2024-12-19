import argparse
import os
import shutil
import torch.multiprocessing
from torch.optim.lr_scheduler import MultiStepLR
import math

from avalanche.benchmarks import nc_benchmark, SplitCIFAR10, SplitCIFAR100,SplitTinyImageNet
from avalanche.evaluation.metrics import loss_metrics, forgetting_metrics
from avalanche.evaluation.multilabel_metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger, CSVLogger, TextLogger, TensorboardLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin, EarlyStoppingPluginForTrainMetric,OneHotPlugin
from examples.utils.utils import fix_seeds
from examples.utils.utils import set_gpu_device
from avalanche.models.simple_cnn import ResNet34,ResNet18
from tools.SAM.solver.build import build_optimizer
from tools.myargs import get_args
from avalanche.evaluation.multilabel_metrics.multiLabelLoss import ReweightedLoss
from tools.class_order import *

torch.multiprocessing.set_sharing_strategy('file_system')

def select_multi_class_dataset(args):
    if args.dataset_name == "cifar10":
        total_class_num = 10
        scenario = SplitCIFAR10(
            n_experiences=1,
            seed=args.seed,
            fixed_class_order=cifar10_class_order,
            balance=False
        )
        model = ResNet18(num_classes=total_class_num)
    elif args.dataset_name == "cifar100":
        total_class_num = 100
        scenario = SplitCIFAR100(
            n_experiences=1,
            seed=args.seed,
            fixed_class_order=cifar100_class_order,
            balance=False
        )
        model = ResNet34(num_classes=total_class_num)
    elif args.dataset_name == "tinyimagenet":
        total_class_num = 200
        scenario = SplitTinyImageNet(
            n_experiences=1,
            seed=args.seed,
            fixed_class_order=tinyimagenet_class_order,
            balance=False
        )
        model = ResNet34(num_classes=total_class_num)
    else:
        raise NotImplementedError
    return total_class_num,scenario,model


def main(args):
    fix_seeds(args.seed)
    device = set_gpu_device(args.cuda)

    total_class_num, scenario, model = select_multi_class_dataset(args)

    if os.path.isdir(args.csvlog_dir):
        shutil.rmtree(args.csvlog_dir)

    #  choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    t_logger = TensorboardLogger()
    csv_logger = CSVLogger(log_folder=args.csvlog_dir)
    txt_logger = TextLogger(open(args.csvlog_dir + "/" + args.txtlog_name, 'a+'))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # forward_transfer_metrics(experience=True, stream=True),
        loggers=[interactive_logger, csv_logger,t_logger, txt_logger],
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
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    parser.add_argument('--dataset_name',type=str,default='cifar100')
    parser.add_argument('--crit',type=str,default='BCE',help="Option between BCE and Reweighted")

    args = get_args(parser)
    print(args)
    main(args)
