from typing import Any, cast

import pandas as pd
from ucimlrepo import fetch_ucirepo


class Dataset:
    def __init__(self):
        self.targets: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None

    def load_from_uci(self, dataset_name: str | None, dataset_id: int | None):

        if dataset_name is not None and dataset_id is None:
            ds = fetch_ucirepo(name=dataset_name)
        elif dataset_name is None and dataset_id is not None:
            ds = fetch_ucirepo(id=dataset_id)
        else:
            raise Exception(
                "load_from_uci: You must specify a dataset name OR dataset id."
            )

        data: Any = ds

        self.features = data.data.features
        self.targets = data.data.targets
