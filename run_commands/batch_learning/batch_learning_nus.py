from run_commands.batch_learning.usage import naive_usage
common_args = {
    'batch_size':196,
    'learning_rate':0.005,
    'num_epochs':50,
    'cuda':'1',
    'num_workers':4
}
if __name__ == '__main__':
    # naive_usage(
    #     des="",
    #     dataset_name="nus",
    #     crit="BCE",
    #     py_file="multi_label_batch_learning.py",
    #     seed_list=[2222],
    #     epoch=common_args["num_epochs"],
    #     batch_size=common_args["batch_size"],
    #     opt="sgd",
    #     cu=common_args['cuda'],
    #     lr=common_args['learning_rate'],
    #     num_workers=common_args['num_workers'],
    #     Cs=[0.7]
    # )
    naive_usage(
        des="",
        dataset_name="nus",
        crit="R",
        py_file="multi_label_batch_learning.py",
        seed_list=[2222],
        epoch=common_args["num_epochs"],
        batch_size=common_args["batch_size"],
        opt="sgd",
        cu=common_args['cuda'],
        lr=common_args['learning_rate'],
        num_workers=common_args['num_workers'],
        Cs=[0.7]
    )