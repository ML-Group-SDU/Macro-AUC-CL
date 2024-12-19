################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-04-2022                                                             #
# Author(s): Florian Mies                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Replay strategy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop
import torch.optim.lr_scheduler
from avalanche.benchmarks import SplitCIFAR10
from avalanche.models import SimpleMLP, SimpleCNN, MlpVAE, VAE_loss
from avalanche.training import VAETraining
from avalanche.training.supervised import GenerativeReplay
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, GenerativeReplayPlugin


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- SCENARIO CREATION
    scenario = SplitCIFAR10(n_experiences=10, seed=1234)
    # ---------

    # MODEL CREATION

    model = SimpleCNN(num_classes=scenario.n_classes)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    generator = MlpVAE((3, 32, 32), nhid=5, device=device)
    # optimzer:

    train_mb_size = 100
    train_epochs = 4
    eval_mb_size = 100

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    optimizer_generator = torch.optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=0.01,
        weight_decay=0.0001,
    )
    # strategy (with plugin):
    generator_strategy = VAETraining(
        model=generator,
        optimizer=optimizer_generator,
        criterion=VAE_loss,
        train_mb_size=train_mb_size,
        train_epochs=10,
        eval_mb_size=eval_mb_size,
        device=device,
        plugins=[
            GenerativeReplayPlugin(
                replay_size=2000,
                increasing_replay_size=True,
            )
        ],
    )

    cl_strategy = GenerativeReplay(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=15,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
        generator_strategy=generator_strategy
    )



    # TRAINING LOOP
    print("Starting experiment...")
    results = []

    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(scenario.test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=1,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
