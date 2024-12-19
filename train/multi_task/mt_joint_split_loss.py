import os
import sys

sys.path.insert(0, "/home/zhangyan/codes/MACRO-AUC-CL/")

from train.utils.my_tools import *

import shutil
import torch.multiprocessing
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.evaluation.multilabel_metrics.multilabel_accuracy import macro_auc_metrics
from avalanche.evaluation.multilabel_metrics.multi_label_forgetting_bwt import forgetting_metrics
from avalanche.logging import InteractiveLogger, CSVLogger, TextLogger
from avalanche.training.supervised.strategy_wrappers_multi_label import MultiTask
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from examples.utils.utils import fix_seeds
from examples.utils.utils import set_gpu_device

from tools.SAM.solver.build import build_optimizer
from tools.myargs import get_args
from avalanche.evaluation.multilabel_metrics.multiLabelLoss import choose_loss

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
    else:
        raise ValueError("Metrics is Wrong!")

    eval_plugin = EvaluationPlugin(
        measure_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        # forgetting_metrics(experience=True, stream=True),
        # forward_transfer_metrics(experience=True, stream=True),
        loggers=[interactive_logger, txt_logger],
    )

    optim, base_optim = build_optimizer(args, model=model)

    if args.opt == "sgd":
        sched = LRSchedulerPlugin(MultiStepLR(optim, [15, 25], gamma=0.4))
        plugin = [sched]
    else:
        plugin = []

    # loss function
    crit = choose_loss(args)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = MultiTask(
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
        scale_ratio=args.scale_ratio
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    if args.eval_on_random:
        cl_strategy.eval(scenario.test_stream)

    experiences = [task for task in scenario.train_stream]
    cl_strategy.train(experiences, num_workers=args.num_workers)
    print("Training completed")

    print("Computing accuracy on the whole test set")
    results.append(cl_strategy.eval(scenario.test_stream, num_workers=args.num_workers))
    # torch.save(cl_strategy.model.state_dict(), args.csvlog_dir + f"/model-{t}.pt")


if __name__ == '__main__':
    args = get_args()
    args.lr = 0.01
    print(args)
    main(args)
