
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import os
import shutil

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import JointTraining
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger,TextLogger,CSVLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):

    # Config
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # model
    model = SimpleMLP(num_classes=10)

    # CL Benchmark Creation
    perm_mnist = PermutedMNIST(n_experiences=5,seed=args.seed)
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream

    # Prepare for training & testing
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001,)
    criterion = CrossEntropyLoss()


    if os.path.isdir(args.csvlog_dir):
        shutil.rmtree(args.csvlog_dir)
    interactive_logger = InteractiveLogger()
    csv_logger = CSVLogger(log_folder=args.csvlog_dir)
    txt_logger = TextLogger(open(args.csvlog_dir + "/" + args.txtlog_name, 'a+'))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger,txt_logger],
    )

    # Joint training strategy
    joint_train = JointTraining(
        model,
        optimizer,
        criterion,
        train_mb_size=32,
        train_epochs=1,
        eval_mb_size=32,
        device=device,
        plugins=[eval_plugin],
    )

    # train and test loop
    results = []
    print("Starting training.")
    # Differently from other avalanche strategies, you NEED to call train
    # on the entire stream.
    joint_train.train(train_stream)
    results.append(joint_train.eval(test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda",type=int,default=0,help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    args = parser.parse_args()

    main(args)
