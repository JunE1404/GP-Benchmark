from .regression_dataset import GetLocal, RegressionDataset
import pandas as pd

class UCIProtein(RegressionDataset):
    def __init__(self) -> None:
        """Load the UCI Protein Tertiary Structure dataset (id=265).

        Uses the local cache when present, otherwise reads
        ``src/datasets/localfiles/CASP.csv``; column 0 is the target and the
        remaining columns are continuous features.
        """
        f_local, t_local = GetLocal(self)
        if f_local is None or t_local is None:
            data = pd.read_csv("src/datasets/localfiles/CASP.csv")
            features = data.iloc[:, 1:].to_numpy()
            targets = data.iloc[:, 0].to_numpy()
        else:
            features = f_local
            targets = t_local
        feature_types = ["con"] * features.shape[1]
        super().__init__(features, targets, feature_types)
