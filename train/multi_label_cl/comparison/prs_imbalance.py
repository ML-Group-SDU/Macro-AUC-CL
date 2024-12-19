import os
import sys
sys.path.insert(0, "/home2/zhangyan/codes/MACRO-AUC-CL/")
import shutil
import torch.multiprocessing
from torch.optim.lr_scheduler import MultiStepLR
from train.utils.my_tools import *

from avalanche.evaluation.metrics import accuracy_metrics,loss_metrics
from avalanche.evaluation.multilabel_metrics.multilabel_accuracy import macro_auc_metrics
from avalanche.evaluation.multilabel_metrics.multilabel_map import map_metrics
from avalanche.evaluation.multilabel_metrics.multi_label_forgetting_bwt import forgetting_metrics
from avalanche.logging import InteractiveLogger,CSVLogger,TextLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin, PRSPlugin
from examples.utils.utils import fix_seeds
from examples.utils.utils import set_gpu_device
from tools.SAM.solver.build import build_optimizer
from tools.myargs import get_args

torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    fix_seeds(args.seed)
    device_ids = [int(ele) for ele in args.cuda.split(",")]
    device = set_gpu_device(device_ids[0])

    scenario, total_class_num = get_benchmark(args)
    model = select_model(args.model, total_class_num)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    if os.path.isdir(args.csvlog_dir):
        shutil.rmtree(args.csvlog_dir)

    #  choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    #  t_logger = TensorboardLogger()
    csv_logger = CSVLogger(log_folder=args.csvlog_dir)
    txt_logger = TextLogger(open(args.csvlog_dir + "/" + args.txtlog_name, 'a+'))

    if args.measure == "macro-auc":
        measure_metrics = macro_auc_metrics
    elif args.measure == "acc":
        measure_metrics = accuracy_metrics
    elif args.measure == "map":
        measure_metrics = map_metrics
    else:
       raise ValueError("Metrics is Wrong!")

    eval_plugin = EvaluationPlugin(
        measure_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # forward_transfer_metrics(experience=True, stream=True),
        loggers=[interactive_logger, csv_logger, txt_logger],
    )

    optim, base_optim = build_optimizer(args, model=model)
    if args.opt == "sgd":
        sched = LRSchedulerPlugin(
            MultiStepLR(optim, [15, 25], gamma=0.2)
        )
        plugin = [sched, PRSPlugin(mem_size=args.mem_size)]
    else:
        plugin = [PRSPlugin(mem_size=args.mem_size)]

    # loss function

    crit = torch.nn.MultiLabelSoftMarginLoss()


    # CREATE THE STRATEGY INSTANCE (NAIVE)

    cl_strategy = Naive(
        model,
        optim,
        criterion=crit,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=args.batch_size,
        device=device,
        plugins=plugin,
        evaluator=eval_plugin,
        do_initial=args.do_initial,
        eval_every=0,
        use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'),
    )

    # TRAINING LOOP
    train_loop(args, cl_strategy, scenario)

if __name__ == '__main__':
    args = get_args()

    print(args)
    main(args)
