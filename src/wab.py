import random
from typing import List, Tuple

import wandb
from scaffolds import WandBDetails, RunArguments, LogDetails, RunSummary
from dataclasses import asdict
from pathlib import Path
import pandas as pd


class WandBRun():
    def __init__(self, wandbDetails: WandBDetails, runDetails: RunArguments, log_path: Path):
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=wandbDetails.entity,
            # Set the wandb project where this run will be logged.
            project=wandbDetails.project,
            name=wandbDetails.name,
            # Track hyperparameters and run metadata.
            config=runDetails
        )   
        self.log_path = log_path

    def get_run (self):
        return self

    def log(self, logDetails: LogDetails):
        details= asdict(logDetails)
        self.run.log(details)  
        df = pd.DataFrame([details])
        first_write = not self.log_path.exists()
        df.to_csv(self.log_path, mode='a', index=False, header=first_write)

    def summarize(self, summary: RunSummary):
        self.run.summary.update(asdict(summary))

    def finish(self):
        self.run.finish()

