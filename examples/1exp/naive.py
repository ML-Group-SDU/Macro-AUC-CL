import torch
from os.path import expanduser

"""
A simple example on how to use the Naive strategy.
"""

from avalanche.models import ResNet34
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.classic import SplitTinyImageNet
from avalanche.logging import InteractiveLogger
from avalanche.training.supervised import (
    Naive
)


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # create the benchmark
    benchmark = SplitTinyImageNet(
        n_experiences=1,
        dataset_root=expanduser("~") + "/.avalanche/data/tinyimagenet/"
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
    )

    model = ResNet34(num_classes=200)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # create strategy
    strategy = Naive(
        model,
        optimizer,
        criterion,
        train_epochs=100,
        device=device,
        train_mb_size=128,
        evaluator=eval_plugin,
    )

    # train on the selected benchmark with the chosen strategy
    for experience in benchmark.train_stream:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)
        strategy.eval(benchmark.test_stream[:])


if __name__ == "__main__":
    main()
