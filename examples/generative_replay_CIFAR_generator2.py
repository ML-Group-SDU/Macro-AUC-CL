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
import os
import pickle

import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from avalanche.benchmarks import SplitCIFAR10
from avalanche.evaluation.metrics import EpochLoss,ExperienceLoss,MinibatchLoss
from avalanche.training.supervised import VAETraining
from avalanche.training.plugins import GenerativeReplayPlugin, EvaluationPlugin, LRSchedulerPlugin
from avalanche.logging import InteractiveLogger
from avalanche.models import RES_VAE,loss_func
from torch.optim.lr_scheduler import MultiStepLR

def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else None
    )
    transform = transforms.Compose([transforms.Resize([224, 224]),
                                    transforms.ToTensor()])  # for grayscale images

    # --- SCENARIO CREATION
    scenario = SplitCIFAR10(n_experiences=1, seed=1234, train_transform=transform, eval_transform=transform)
    # ---------

    # MODEL CREATION
    model = RES_VAE(z_dim=256).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, milestones=[14, 23], gamma=0.2)
    )

    evaluator = EvaluationPlugin(
        EpochLoss(),
        ExperienceLoss(),
        loggers=[InteractiveLogger()],
    )
    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = VAETraining(
        model,
        optim,
        train_mb_size=100,
        train_epochs=35,
        device=device,
        criterion=loss_func,
        plugins=[GenerativeReplayPlugin(),sched],
        evaluator=evaluator,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    f, axarr = plt.subplots(scenario.n_experiences, 10)
    k = 0
    if scenario.n_experiences == 1:
        axarr=axarr[np.newaxis,:]

    trans = transforms.Compose([
        transforms.ToPILImage()
    ])
    g_batch_size = 10
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        with open("./imgs/model.pkl", 'wb') as file:
            pickle.dump(model, file)

        samples = model.generate(g_batch_size)
        # samples = samples.detach().cpu().numpy()

        for j in range(g_batch_size):
            image = trans(samples[j])
            image.save("./imgs/{}_{}.jpg".format(k,j))
            axarr[k, j].imshow(np.array(image))
            axarr[k, 4].set_title("Generated images for experience " + str(k))
        np.vectorize(lambda ax: ax.axis("off"))(axarr)
        k += 1

    f.subplots_adjust(hspace=1.2)
    plt.savefig("VAE_output_per_exp")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=2,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
