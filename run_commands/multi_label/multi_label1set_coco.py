import sys
sys.path.insert(0,"/home/zhangyan/codes/avalanche-project/")

from usage import multi_task_for1dataset_usage

if __name__ == '__main__':
    # multi_task_for1dataset_usage(
    #     dataset_name="coco",
    #     des='',
    #     task_num=8,
    #     crit="BCE",
    #     measure='macro-auc',
    #     py_file="multi_label.py",
    #     seed_list=[2222],
    #     epoch=35,
    #     batch_size=128,
    #     opt="sgd",
    #     cu=0,
    #     lr=0.01,
    # )
    multi_task_for1dataset_usage(
        dataset_name="coco",
        des='',
        task_num=8,
        crit="ReWeighted",
        measure='macro-auc',
        py_file="multi_label.py",
        seed_list=[2222],
        epoch=35,
        batch_size=128,
        opt="sgd",
        cu=0,
        lr=0.01,
    )

