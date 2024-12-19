from run_commands.multi_label.usage import run_ml_cl_tasks_from_single_dataset

voc_setup = {
    'dataset': 'voc',
    'task_num': 5,
    'cuda': '2'
}
coco_setup = {
    'dataset': 'coco',
    'task_num': 8,
}
nus_setup = {
    'dataset': 'nus-wide',
    'task_num': 9,
}
setup = coco_setup

if __name__ == '__main__':
    run_ml_cl_tasks_from_single_dataset(
        dataset_name=setup["dataset"],
        task_num=setup["task_num"],
        des="scale_loss",
        crit="RM",
        measure='macro-auc',
        py_file="multi_label_cl/rm_wrs.py",
        seed_list=[2222],
        epoch=40,
        batch_size=256,
        opt="sgd",
        cu="0",
        lr=0.01,
        model="res34",
        imbalance="yes",
        mem_sizes=[2000],
        lambdas=[3.5],
        num_workers=2,
        scale_ratio_list=[1.0, 0.8, 0.6]
    )
