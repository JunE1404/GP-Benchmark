import random
from typing import List, Tuple

import wandb
from scaffolds import WandBDetails, RunArguments, LogDetails, RunSummary
from dataclasses import asdict
from pathlib import Path
import pandas as pd


class WandBRun():
    """Thin wrapper around a Weights & Biases run that also mirrors logs to CSV.

    Args:
        wandbDetails: Entity, project and run name for the W&B run.
        runDetails: Run configuration stored as the W&B run config.
        log_path: CSV path that per-iteration logs are appended to.
    """

    def __init__(self, wandbDetails: WandBDetails, runDetails: RunArguments, log_path: Path) -> None:
        """Initialize the W&B run and remember the CSV log path."""
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

    def get_run (self) -> WandBRun:
        """Return this wrapper instance."""
        return self

    def log(self, logDetails: LogDetails) -> None:
        """Log one iteration to W&B and append it to the CSV file."""
        details= asdict(logDetails)
        self.run.log(details)  
        df = pd.DataFrame([details])
        first_write = not self.log_path.exists()
        df.to_csv(self.log_path, mode='a', index=False, header=first_write)

    def summarize(self, summary: RunSummary) -> None:
        """Write final aggregate metrics to the W&B run summary."""
        self.run.summary.update(asdict(summary))

    def finish(self) -> None:
        """Finish the underlying W&B run."""
        self.run.finish()

