from sklearn.cluster import KMeans
from torch import Tensor
import torch
import numpy as np

def getInducingPoints(data: Tensor, k: int):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(data.numpy())
    return torch.tensor(kmeans.cluster_centers_)
