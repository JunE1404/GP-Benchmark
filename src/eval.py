from datasets.uci_wine import UCIWineQuality
import json
from scaffolds import EvalGroupArguments
from unittest import result
import pandas as pd
from os import listdir, walk
from os.path import isfile, join, isdir
import numpy as np



def getFilePathes(criteria: EvalGroupArguments, dir: str):
    files_ret = []
    group_names = []
    for ds in criteria.dataset:
        local_ret = []
        for root, subdirs, files in walk(dir):
            for filename in files: 
                if filename.endswith(".json"):
                    with open(join(root, filename), "r") as f:
                        data = json.load(f)
                        add = True
                        if (ds != None) and (data["dataset"] != str(ds)):
                            add = False
                        if (criteria.gp != None) and (data["modelType"] != criteria.gp):
                            add = False
                        if (criteria.kernel != None) and (data["kernel"] != criteria.kernel):
                            add = False 
                        if (criteria.optimizer != None) and (data["optimizer"] != criteria.optimizer):
                            add = False
                        if (criteria.seed != None) and (data["seed"] != criteria.seed):
                            add = False
                        if (criteria.svgp_strategy != None) and (criteria.svgp_strategy not in data.get("modelType", "")):
                            add = False
                        if (criteria.train_signal_variance != None) and (data["trained_output_scale"] != criteria.train_signal_variance):
                            add = False 
                        
                        filename_log = filename.removesuffix(".json")+".csv"
                        to_add = [join(root, filename), join(root,"logs", filename_log)]

                        if add:
                            local_ret.append(to_add)
        files_ret.append(local_ret)
        group_names.append(_group_stem(criteria, ds))
    return files_ret, group_names


def _group_stem(criteria: EvalGroupArguments, ds) -> str:
    parts = [str(ds)]
    if criteria.gp is not None:
        parts.append(criteria.gp)
    if criteria.kernel is not None:
        parts.append(criteria.kernel)
    if criteria.optimizer is not None:
        parts.append(criteria.optimizer)
    if criteria.train_signal_variance is not None:
        parts.append("OSTrained" if criteria.train_signal_variance else "OSNotTrained")
    if criteria.seed is not None:
        parts.append(f"seed{criteria.seed}")
    if criteria.svgp_strategy is not None:
        parts.append(criteria.svgp_strategy)
    return "_".join(parts)


def meanEvalData(filepathes):
    MAEs = []
    NLLs = []
    PICP50s = []
    PICP90s = []
    PICP95s = []
    RMSEs = []
    Lengthscales = []
    for p in filepathes:
        if len(p) > 1:
            p = p[0]
        with open(p, "r") as f:
            data = json.load(f)
            evalData = data["evalData"]
            MAEs.append(evalData["MAE"])
            NLLs.append(evalData["NLL"])
            PICP50s.append(evalData["PICP50"])
            PICP90s.append(evalData["PICP90"])
            PICP95s.append(evalData["PICP95"])
            RMSEs.append(evalData["RMSE"])
            Lengthscales.append(evalData["Lengthscale"])

    return {
        "MAE": np.mean(MAEs),
        "NLL": np.mean(NLLs),
        "PICP50": np.mean(PICP50s),
        "PICP90": np.mean(PICP90s),
        "PICP95": np.mean(PICP95s),
        "RMSE": np.mean(RMSEs),
        "Lengthscale": np.mean(Lengthscales)
    }

def meanLogs(filepathes, outnames):
    dataset_csv_paths = []
    for i, ds in enumerate(filepathes):
        dataframes = []
        for p in ds:
            if len(p) > 1:
                p = p[1]
            data = pd.read_csv(p)
            dataframes.append(data)

        meandf =pd.concat(dataframes, axis=0).dropna(axis=1).groupby('iteration').mean()
        meandf.to_csv(f"eval/{outnames[i]}.csv")
        dataset_csv_paths.append(f"eval/{outnames[i]}.csv")
    return dataset_csv_paths


def methodRunData(path):
    data = pd.read_csv(path)
    mean_rmse = data.min(axis=0)['val_RMSE']
    mean_nll = data.min(axis=0)['val_NLL']
    std_rmse = data['val_RMSE'].std()
    std_nll = data['val_NLL'].std()
    mean_runtime_clean = data["it_time_training"].sum()
    return mean_rmse, mean_nll, mean_runtime_clean, std_rmse, std_nll
