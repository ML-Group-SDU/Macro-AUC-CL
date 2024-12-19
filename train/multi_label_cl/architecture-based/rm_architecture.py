import os
import sys
sys.path.insert(0,"/home/zhangyan/codes/avalanche-project/")
import shutil
import torch.multiprocessing
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.evaluation.metrics import accuracy_metrics,loss_metrics
from avalanche.evaluation.multilabel_metrics.multilabel_accuracy import macro_auc_metrics
from avalanche.evaluation.multilabel_metrics.multi_label_forgetting_bwt import forgetting_metrics
from avalanche.logging import InteractiveLogger,CSVLogger,TextLogger
from avalanche.training import MultiLabelNaive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin

from avalanche.models.vgg import SingleHeadVGGSmall
from avalanche.models.packnet import PackNetModel
from examples.utils.utils import fix_seeds
from examples.utils.utils import set_gpu_device

from tools.SAM.solver.build import build_optimizer
from tools.myargs import get_args
from avalanche.evaluation.multilabel_metrics.multiLabelLoss import ReweightedMarginLoss,SpecficBCELoss,ReweightedLoss

torch.multiprocessing.set_sharing_strategy('file_system')
from avalanche.models.packnet import PackNetPlugin
def main(args):
    fix_seeds(args.seed)
    device_ids = [int(ele) for ele in args.cuda.split(",")]
    device = set_gpu_device(device_ids[0])
    prune_propotion_voc = [
        0.7,
        0.6,
        0.5,
        0.0,
    ]
    prune_propotion_coco = [
        0.87,
        0.85,
        0.83,
        0.80,
        0.75,
        0.66,
        0.50,
        0.00,
    ]
    if args.dataset_name == "voc":
        prune_proportion = prune_propotion_voc
    elif args.dataset_name == "coco":
        prune_proportion = prune_propotion_coco

    scenario, total_class_num = get_benchmark(args)
    # scenario = SplitTinyImageNet(
    #     10, return_task_id=True,
    # )
    model = SingleHeadVGGSmall(n_classes=total_class_num)
    # model = select_model(args.model,total_class_num)
    model = PackNetModel(model)
    # model = torch.nn.DataParallel(model, device_ids=device_ids)

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
        forgetting_metrics(experience=True, stream=True),
        # forward_transfer_metrics(experience=True, stream=True),
        loggers=[interactive_logger, csv_logger, txt_logger],
    )

    optim, base_optim = build_optimizer(args, model=model)
    pack_plugin = PackNetPlugin(15, prune_proportion=prune_proportion)
    if args.opt == "sgd":
        sched = LRSchedulerPlugin(
            MultiStepLR(optim, [15, 25], gamma=0.2)
        )
        plugin = [sched,pack_plugin]
    else:
        plugin = [pack_plugin]

    # loss function
    print(args.crit)
    if args.crit == "BCE": # BCE without or with WRS
        crit = SpecficBCELoss()
    elif args.crit == "R":
        crit = ReweightedLoss()
    elif args.crit == "RM":
        crit = ReweightedMarginLoss(C=args.C,nth_power=args.nth_power)
    else:
        raise NotImplementedError

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = MultiLabelNaive(
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
    train_loop(args,cl_strategy,scenario)

if __name__ == '__main__':
    args = get_args()

    print(args)
    main(args)
