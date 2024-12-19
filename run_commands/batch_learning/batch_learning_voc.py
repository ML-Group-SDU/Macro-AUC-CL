from run_commands.batch_learning.usage import naive_usage
common_args = {
    'batch_size':192,
    'learning_rate':0.005,
    'num_epochs':50,
    'cuda':3,
    'num_workers':2
}
if __name__ == '__main__':
    # naive_usage(
    #     dataset_name="voc",
    #     crit="BCE",
    #     py_file="multi_label_batch_learning.py",
    #     seed_list=[2222],
    #     epoch=common_args["num_epochs"],
    #     batch_size=common_args["batch_size"],
    #     opt="sgd",
    #     cu=common_args['cuda'],
    #     lr=common_args['learning_rate'],
    #     num_workers=common_args['num_workers'],
    #     Cs=[4]
    # )
    naive_usage(
        des='',
        dataset_name="voc",
        crit="RM",
        py_file="multi_label_batch_learning.py",
        seed_list=[2222],
        epoch=common_args["num_epochs"],
        batch_size=common_args["batch_size"],
        opt="sgd",
        cu=common_args['cuda'],
        lr=common_args['learning_rate'],
        num_workers=common_args['num_workers'],
        Cs=[1,1.2,1.4,1.6,1.8]
    )