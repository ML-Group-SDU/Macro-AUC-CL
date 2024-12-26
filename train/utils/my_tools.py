from avalanche.models.simple_cnn import ResNet50, ResNet101, ResNet34
from benchmarks.multi_label_dataset import cl_ml_imbalance_benchmark_from_single_dataset,cl_ml_benckmark_from_single_dataset
import torch

def select_model(model_name,total_class_num):
    if model_name == "res34":
        model = ResNet34(num_classes=total_class_num)
    elif model_name == "res50":
        model = ResNet50(num_classes=total_class_num)
    elif model_name == "res101":
        model = ResNet101(num_classes=total_class_num)
    else:
        raise ValueError("model selection error!")
    return model


def get_benchmark(args):
    if args.imbalance == "yes":
        scenario,total_class_num = cl_ml_imbalance_benchmark_from_single_dataset(dataset_name=args.dataset_name,
                                                                         task_num=args.task_num,use_aug=args.use_aug)
    elif args.imbalance == "no":
        scenario, total_class_num = cl_ml_benckmark_from_single_dataset(dataset_name=args.dataset_name,
                                                                          task_num=args.task_num, use_aug=args.use_aug)
    else:
        raise NotImplementedError
    return scenario,total_class_num


def train_loop(args,cl_strategy,scenario):
    print("Starting experiment...")
    results = []
    if args.eval_on_random:
        cl_strategy.eval(scenario.test_stream)

    for t, experience in enumerate(scenario.train_stream):
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience,[scenario.test_stream[0]],num_workers=args.num_workers)
        print("Training completed")
        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(scenario.test_stream, num_workers=args.num_workers))
        torch.save(cl_strategy.model.state_dict(), args.csvlog_dir + f"/model-{t}.pt")

