import sys
sys.path.insert(0,"/home/zhangyan/codes/avalanche-project/")
from usage import multi_task_for1dataset_usage

if __name__ == '__main__':
    # multi_task_for1dataset_usage(
    #     dataset_name="nus-wide",
    #     task_num=9,
    #     des='',
    #     crit="BCE",
    #     measure='macro-auc',
    #     py_file="multi_label.py",
    #     seed_list=[3088],
    #     epoch=35,
    #     batch_size=128,
    #     opt="sgd",
    #     cu=1,
    #     lr=0.01,
    # )
    multi_task_for1dataset_usage(
        dataset_name="nus-wide",
        task_num=9,
        des='',
        crit="ReWeighted",
        measure='macro-auc',
        py_file="multi_label.py",
        seed_list=[3088],
        epoch=35,
        batch_size=128,
        opt="sgd",
        cu=1,
        lr=0.01,
    )

