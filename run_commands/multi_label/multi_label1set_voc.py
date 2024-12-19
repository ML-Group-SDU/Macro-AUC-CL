from run_commands.multi_label.usage import multi_task_for1dataset_usage

if __name__ == '__main__':
    multi_task_for1dataset_usage(
        dataset_name="voc",
        task_num=5,
        des="ntt",
        crit="BCE",
        measure='macro-auc',
        py_file="multi_label.py",
        seed_list=[2222],
        epoch=40,
        batch_size=128,
        opt="sgd",
        cu=3,
        lr=0.01,
    )
    multi_task_for1dataset_usage(
        dataset_name="voc",
        task_num=5,
        des="ntt",
        crit="ReWeighted",
        measure='macro-auc',
        py_file="multi_label.py",
        seed_list=[2222],
        epoch=40,
        batch_size=128,
        opt="sgd",
        cu=3,
        lr=0.01,
        use_aug="no"
    )

