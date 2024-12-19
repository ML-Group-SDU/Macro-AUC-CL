from run_commands.multi_label.usage import multi_task_for1dataset_usage

if __name__ == '__main__':
    des = "ocdm/voc_imb_34"

    multi_task_for1dataset_usage(
        dataset_name="voc",
        task_num=5,
        des=des+"_1000",
        crit="MultiSoft",
        measure='macro-auc',
        py_file="ocdm_imbalance.py",
        seed_list=[2222],
        epoch=40,
        batch_size=128,
        opt="sgd",
        cu='0',
        lr=0.01,
        use_aug="no",
        mem_size=1000,
        model="res34",
        imbalance="yes"
    )
    multi_task_for1dataset_usage(
        dataset_name="voc",
        task_num=5,
        des=des + "_500",
        crit="MultiSoft",
        measure='macro-auc',
        py_file="ocdm_imbalance.py",
        seed_list=[2222],
        epoch=40,
        batch_size=128,
        opt="sgd",
        cu='0',
        lr=0.01,
        use_aug="no",
        mem_size=500,
        model="res34",
        imbalance="yes"
    )



