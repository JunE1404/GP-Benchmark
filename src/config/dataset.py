
from datasets.regression_dataset import RegressionDataset
from datasets.synthetic_simple import SimpleSyntheticDataset
from datasets.uci_parkinsons import UCIParkinsonsTelemonitoring
from datasets.uci_proteins import UCIProtein
from datasets.uci_wine import UCIWineQuality
from datasets.uci_keggu import UCIKeggu
from datasets.uci_road import UCIRoad
from misc.scaffolds import RunArguments
def getDataset(arguments: RunArguments) -> RegressionDataset | None:
    """Instantiate the dataset named by ``arguments.dataset``.

    Args:
        arguments: Run configuration. ``arguments.dataset`` selects the concrete
            dataset class.

    Returns:
        The matching :class:`RegressionDataset` instance, or ``None`` when the
        name matches no known dataset.

    Raises:
        ValueError: If ``arguments.dataset`` is ``None``.
    """
    if arguments.dataset is None:
        raise ValueError("No dataset argument provided.")
    dset = None
    match arguments.dataset:
        case "synth":
            dset = SimpleSyntheticDataset()
        case "parkinsons":
            dset = UCIParkinsonsTelemonitoring()
        case "wine":
            dset = UCIWineQuality()
        case "protein":
            dset = UCIProtein()
        case "road":
            dset = UCIRoad()
        case "keggu":
            dset = UCIKeggu()
    return dset