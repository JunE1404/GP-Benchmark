import random

import wandb
from scaffolds import WandBDetails, RunArguments, LogDetails
from dataclasses import asdict


class WandBRun():
    def __init__(self, wandbDetails: WandBDetails, runDetails: RunArguments):
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=wandbDetails.entity,
            # Set the wandb project where this run will be logged.
            project=wandbDetails.project,
            name=wandbDetails.name,
            # Track hyperparameters and run metadata.
            config=runDetails
        )

    def log(self, logDetails: LogDetails):
        self.run.log(asdict(logDetails))

    def finish(self):
        self.run.finish()