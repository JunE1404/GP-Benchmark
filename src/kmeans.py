from sklearn.cluster import KMeans
from torch import Tensor
import torch
import numpy as np

def getInducingPoints(data: Tensor, k: int, strategy: str, seed: int):
    match strategy:
        case "kmeans":
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(data.numpy())
            return torch.tensor(kmeans.cluster_centers_)
        case "random":
            #todo randomly select k-element subset of data
            pass
