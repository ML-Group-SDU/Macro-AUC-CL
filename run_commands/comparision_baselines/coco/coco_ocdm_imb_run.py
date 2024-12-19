from run_commands.multi_label.usage import multi_task_for1dataset_usage

if __name__ == '__main__':
    des = "ocdm/coco_imb_50"

    multi_task_for1dataset_usage(
        dataset_name="coco",
        task_num=8,
        des=des+"_2000",
        crit="MultiSoft",
        measure='macro-auc',
        py_file="ocdm_imbalance.py",
        seed_list=[2222],
        epoch=35,
        batch_size=240,
        opt="sgd",
        cu='0,1,2,3,4',
        lr=0.01,
        use_aug="no",
        mem_size=2000,
        model="res50",
        imbalance="yes"
    )
    # multi_task_for1dataset_usage(
    #     dataset_name="coco",
    #     task_num=8,
    #     des=des + "_1500",
    #     crit="MultiSoft",
    #     measure='macro-auc',
    #     py_file="ocdm_imbalance.py",
    #     seed_list=[2222],
    #     epoch=40,
    #     batch_size=96,
    #     opt="sgd",
    #     cu='0,1',
    #     lr=0.01,
    #     use_aug="no",
    #     mem_size=1500,
    #     model="res50",
    #     imbalance="yes"
    # )



