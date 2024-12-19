import os
import shutil
import torch.multiprocessing
from torch.optim.lr_scheduler import MultiStepLR

from benchmarks.multi_label_dataset import multi_label_batchlearning_benchmark
from avalanche.evaluation.multilabel_metrics import loss_metrics
from avalanche.evaluation.multilabel_metrics.multilabel_accuracy import macro_auc_metrics
from avalanche.evaluation.multilabel_metrics.multi_label_forgetting_bwt import forgetting_metrics
from avalanche.logging import InteractiveLogger,CSVLogger,TextLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from examples.utils.utils import fix_seeds
from examples.utils.utils import set_gpu_device
from avalanche.models.simple_cnn import ResNet34
from tools.SAM.solver.build import build_optimizer
from tools.myargs import get_args
from avalanche.evaluation.multilabel_metrics.multiLabelLoss import ForMicroLoss

torch.multiprocessing.set_sharing_strategy('file_system')

def select_multilabel_dataset(dataset_name,seed):
    if dataset_name == "voc":
        total_class_num = 20
    elif dataset_name == "coco":
        total_class_num = 80
    elif dataset_name == "nus_wide":
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

    if os.path.isdir(args.csvlog_dir):
        shutil.rmtree(args.csvlog_dir)

    #  choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    #  t_logger = TensorboardLogger()
    csv_logger = CSVLogger(log_folder=args.csvlog_dir)
    txt_logger = TextLogger(open(args.csvlog_dir + "/" + args.txtlog_name, 'a+'))

    if args.measure == "macro-auc":
        measure_metrics = macro_auc_metrics
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

    sched = LRSchedulerPlugin(
        MultiStepLR(optim, [15, 25], gamma=0.2)
    )

    # loss function
    crit = ForMicroLoss(mode="u_1")

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        optim,
        criterion=crit,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=args.batch_size,
        device=device,
        plugins=[sched],
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
    args = get_args()

    print(args)
    main(args)
