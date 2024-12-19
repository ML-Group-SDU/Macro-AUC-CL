import os
import shutil
import torch.multiprocessing
from torch.optim.lr_scheduler import MultiStepLR
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(PathProject)

from benchmarks.multi_label_dataset import multi_label_batchlearning_benchmark
from avalanche.evaluation.metrics import accuracy_metrics,loss_metrics
from avalanche.evaluation.multilabel_metrics.multilabel_accuracy import macro_auc_metrics
from avalanche.evaluation.multilabel_metrics.multi_label_forgetting_bwt import forgetting_metrics
from avalanche.logging import InteractiveLogger,CSVLogger,TextLogger
from avalanche.training import MultiLabelNaive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from train.utils.my_tools import train_loop
from examples.utils.utils import fix_seeds
from examples.utils.utils import set_gpu_device
from avalanche.models.simple_cnn import ResNet34
from tools.SAM.solver.build import build_optimizer
from tools.myargs import get_args
from avalanche.evaluation.multilabel_metrics.multiLabelLoss import ReweightedLoss,ReweightedMarginLoss

torch.multiprocessing.set_sharing_strategy('file_system')

def select_multilabel_dataset(dataset_name,seed):
    if dataset_name == "voc":
        total_class_num = 20
    elif dataset_name == "coco":
        total_class_num = 80
    elif dataset_name == "nus":
        total_class_num = 81
    else:
        raise NotImplementedError

    scenario = multi_label_batchlearning_benchmark(dataset_name=dataset_name,seed=args.seed)
    return total_class_num,scenario

def main(args):
    fix_seeds(args.seed)
    device_ids = [int(ele) for ele in args.cuda.split(",")]
    device = set_gpu_device(device_ids[0])

    total_class_num, scenario = select_multilabel_dataset(args.dataset_name,args.seed)

    model = ResNet34(num_classes=total_class_num)
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
        measure_metrics = None
        raise ValueError("Metrics is Wrong!")

    eval_plugin = EvaluationPlugin(
        measure_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # forward_transfer_metrics(experience=True, stream=True),
        loggers=[interactive_logger, csv_logger, txt_logger],
    )
    # optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001,)

    optim, base_optim = build_optimizer(args, model=model)

    sched = LRSchedulerPlugin(
        MultiStepLR(optim, [15, 25], gamma=0.2)
    )

    # loss function


    crit1 = ReweightedLoss()

    crit2 = ReweightedMarginLoss(C=args.C,nth_power=args.nth_power)


    # CREATE THE STRATEGY INSTANCE (NAIVE)
    # cl_strategy = Naive(
    #     model,
    #     optim,
    #     criterion=crit,
    #     train_mb_size=args.batch_size,
    #     train_epochs=args.epoch,
    #     eval_mb_size=args.batch_size,
    #     device=device,
    #     plugins=[sched],
    #     evaluator=eval_plugin,
    #     do_initial=args.do_initial,
    #     eval_every=0,
    #     use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'),
    # )
    cl_strategy = MultiLabelNaive(
        model,
        optim,
        criterion=crit1,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=args.batch_size,
        device=device,
        plugins=[sched],
        evaluator=eval_plugin,
        do_initial=args.do_initial,
        eval_every=args.eval_every,
        use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'),
    )
    # TRAINING LOOP
    train_loop(args, cl_strategy, scenario)

    # 2nd step
    cl_strategy = MultiLabelNaive(
        model,
        optim,
        criterion=crit2,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=args.batch_size,
        device=device,
        plugins=[sched],
        evaluator=eval_plugin,
        do_initial=args.do_initial,
        eval_every=args.eval_every,
        use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'),
    )
    # TRAINING LOOP
    train_loop(args,cl_strategy,scenario)

if __name__ == '__main__':
    args = get_args()

    args.opt = "sgd"
    args.lr = 0.005

    print(args)
    main(args)
