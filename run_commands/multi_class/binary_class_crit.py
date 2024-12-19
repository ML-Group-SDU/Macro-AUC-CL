from run_commands.multi_label.usage import binary_classification_usage

if __name__ == '__main__':
    binary_classification_usage(
        crit="BCE",
        py_file="binary_class_batch_learning.py",
        seed_list=[2222],
        epoch=60,
        batch_size=256,
        opt="sgd",
        cu=1,
        lr=0.01,
        weight_decay=0.001,
        two_class_id=[0,1],
        train_scalers="0.1,1",
        test_scalers="1,0.1"
    )
    binary_classification_usage(
        crit="BCE",
        py_file="binary_class_batch_learning.py",
        seed_list=[2222],
        epoch=60,
        batch_size=256,
        opt="sgd",
        cu=1,
        lr=0.01,
        weight_decay=0.0005,
        two_class_id=[0, 1],
        train_scalers="0.1,1",
        test_scalers="1,0.1"
    )

    binary_classification_usage(
        crit="ReWeighted",
        py_file="binary_class_batch_learning.py",
        seed_list=[2222],
        epoch=60,
        batch_size=256,
        opt="sgd",
        cu=1,
        lr=0.01,
        weight_decay=0.001,
        two_class_id=[0, 1],
        train_scalers="0.1,1",
        test_scalers="1,0.1"
    )
    binary_classification_usage(
        crit="ReWeighted",
        py_file="binary_class_batch_learning.py",
        seed_list=[2222],
        epoch=60,
        batch_size=256,
        opt="sgd",
        cu=1,
        lr=0.01,
        weight_decay=0.0005,
        two_class_id=[0, 1],
        train_scalers="0.1,1",
        test_scalers="1,0.1"
    )

