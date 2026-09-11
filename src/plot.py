from datasets.uci_road import UCIRoad
from datasets.uci_keggu import UCIKeggu
from datasets.uci_parkinsons import UCIParkinsonsTelemonitoring
from scaffolds import EvalGroupArguments
from datasets.uci_wine import UCIWineQuality
from datasets.uci_proteins import UCIProtein
from ast import List
import matplotlib.pyplot as plt

from tueplots import figsizes
import pandas as pd
import eval

from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes


# Increase the resolution of all the plots below
plt.rcParams.update({"figure.dpi": 150})
plt.rcParams['text.usetex'] = True



def plotMetricsForGP(run_csv_pathes: List[List[str]],eval_params: List[str],max_iteration=None,col_labels=None, row_labels=None, group_labels=None, image_filename=None):
    params = figsizes.aistats2025_full(nrows=len(eval_params), ncols=len(run_csv_pathes[0]))
    plt.rcParams.update(params)

    plt.rcParams.update(cycler.cycler(color=palettes.tue_ai))

    fig, axes = plt.subplots(nrows=len(eval_params), ncols=len(run_csv_pathes[0]), sharex=True, layout="constrained")
    for x, ol in enumerate(run_csv_pathes):
        for i, rp in enumerate(ol):
            data =pd.read_csv(rp)
            it = data["iteration"]
            if max_iteration:
                it = it[:max_iteration]
            for j, p in enumerate(eval_params):
                pdata = data[p]
                if max_iteration:
                    pdata = pdata[:max_iteration]
                ax = axes[j][i]
                ax.plot(it, pdata, label=group_labels[x] if group_labels else None)
                if j == 0:          # bottom row
                    ax.set_xlabel(col_labels[i] if col_labels else rp)
                    ax.xaxis.set_label_position("top")

                if i == 0:                              # left column
                    ax.set_ylabel(row_labels[j] if row_labels else p)
    
    if group_labels:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=len(group_labels), loc="outside upper center", frameon=True)

    if image_filename:
        plt.savefig(image_filename)
    else:
        plt.show()



exact_lbfgs_sv = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="ExactGP",
    kernel="RBFKeops",
    optimizer="LBFGS_MaxIter_1",
    train_signal_variance=True
)
exact_lbfgs = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="ExactGP",
    kernel="RBFKeops",
    optimizer="LBFGS_MaxIter_1",
    train_signal_variance=False
)

exact_adam_sv = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="ExactGP",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=True
)

exact_adam = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="ExactGP",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=False
)


#--------------- EXACTCG

exactcg_lbfgs_sv = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="ExactGPConjGradients",
    kernel="RBFKeops",
    optimizer="LBFGS_MaxIter_1",
    train_signal_variance=True
)
exactcg_lbfgs = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="ExactGPConjGradients",
    kernel="RBFKeops",
    optimizer="LBFGS_MaxIter_1",
    train_signal_variance=False
)

exactcg_adam_sv = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="ExactGPConjGradients",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=True
)

exactcg_adam = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="ExactGPConjGradients",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=False
)

#-------------- SVGP

svgp_random_sv = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="SVGP_inducing_init_random",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=True
)

svgp_random = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="SVGP_inducing_init_random",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=False
)
svgp_kmeans_sv = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="SVGP_inducing_init_kmeans",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=True
)

svgp_kmeans = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="SVGP_inducing_init_kmeans",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=False
)

#--------CAGP

cagp_sv = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="CAGP",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=True
)

cagp = EvalGroupArguments(
    dataset=[UCIWineQuality(),UCIParkinsonsTelemonitoring(),UCIProtein(), UCIKeggu(),UCIRoad()],
    gp="CAGP",
    kernel="RBFKeops",
    optimizer="Adam",
    train_signal_variance=False
)


pathes_e_lbfgs_sv, dsn_e_lbfgs_sv = eval.getFilePathes(exact_lbfgs_sv, "results")
pathes_e_lbfgs,dsn_e_lbfgs = eval.getFilePathes(exact_lbfgs, "results")
pathes_e_adam_sv,dsn_e_adam_sv = eval.getFilePathes(exact_adam_sv, "results")
pathes_e_adam,dsn_e_adam = eval.getFilePathes(exact_adam, "results")


e_adam_sv_csv_pathes = eval.meanLogs(pathes_e_adam_sv, dsn_e_adam_sv)
e_adam_csv_pathes = eval.meanLogs(pathes_e_adam, dsn_e_adam)
e_lbfgs_sv_csv_pathes = eval.meanLogs(pathes_e_lbfgs_sv, dsn_e_lbfgs_sv)
e_lbfgs_csv_pathes = eval.meanLogs(pathes_e_lbfgs, dsn_e_lbfgs)


plotMetricsForGP([e_adam_sv_csv_pathes,e_adam_csv_pathes,e_lbfgs_sv_csv_pathes,e_lbfgs_csv_pathes], ["loss","outputscale","likelyhood_noise","test_NLL","test_RMSE", "test_PICP95","test_PICP90","test_PICP50"], 
    max_iteration=150,
    col_labels=["Wine", "Protein", "Parkinsons", "Keggu", "Road"],
    row_labels=["Loss",r"$\sigma^2_f$",r"$\sigma^2_n$", "NLL", "RMSE", "PICP95", "PICP90", "PICP50"],
    group_labels=[r"Adam $\sigma^2_f$ T", r"Adam $\sigma^2_f$ F",r"LBFGS $\sigma^2_f$ T", r"LBFGS $\sigma^2_f$ F"],
    image_filename="assets/exact_signal_variance_comparison"
)

#------------------ EXACTCG


pathes_ecg_lbfgs_sv, dsn_ecg_lbfgs_sv = eval.getFilePathes(exactcg_lbfgs_sv, "results")
pathes_ecg_lbfgs,dsn_ecg_lbfgs = eval.getFilePathes(exactcg_lbfgs, "results")
pathes_ecg_adam_sv,dsn_ecg_adam_sv = eval.getFilePathes(exactcg_adam_sv, "results")
pathes_ecg_adam,dsn_ecg_adam = eval.getFilePathes(exactcg_adam, "results")

print(pathes_ecg_adam)
ecg_adam_sv_csv_pathes = eval.meanLogs(pathes_ecg_adam_sv, dsn_ecg_adam_sv)
ecg_adam_csv_pathes = eval.meanLogs(pathes_ecg_adam, dsn_ecg_adam)
ecg_lbfgs_sv_csv_pathes = eval.meanLogs(pathes_ecg_lbfgs_sv, dsn_ecg_lbfgs_sv)
ecg_lbfgs_csv_pathes = eval.meanLogs(pathes_ecg_lbfgs, dsn_ecg_lbfgs)

plotMetricsForGP([ecg_adam_sv_csv_pathes,ecg_adam_csv_pathes,ecg_lbfgs_sv_csv_pathes,ecg_lbfgs_csv_pathes], ["loss","outputscale","likelyhood_noise","test_NLL","test_RMSE", "test_PICP95","test_PICP90","test_PICP50"], 
    max_iteration=150,
    col_labels=["Wine", "Protein", "Parkinsons", "Keggu", "Road"],
    row_labels=["Loss",r"$\sigma^2_f$",r"$\sigma^2_n$", "NLL", "RMSE", "PICP95", "PICP90", "PICP50"],
    group_labels=[r"Adam $\sigma^2_f$ T", r"Adam $\sigma^2_f$ F",r"LBFGS $\sigma^2_f$ T", r"LBFGS $\sigma^2_f$ F"],
    image_filename="assets/exactcg_signal_variance_comparison"
)

#--------- SVGP
pathes_svgp_random_sv, dsn_svgp_random_sv = eval.getFilePathes(svgp_random_sv, "results")
pathes_svgp_random,dsn_svgp_random = eval.getFilePathes(svgp_random, "results")
pathes_svgp_kmeans_sv,dsn_svgp_kmeans_sv = eval.getFilePathes(svgp_kmeans_sv, "results")
pathes_svgp_kmeans,dsn_svgp_kmeans = eval.getFilePathes(svgp_kmeans, "results")

svgp_random_sv_csv_pathes = eval.meanLogs(pathes_svgp_random_sv, dsn_svgp_random_sv)
svgp_random_csv_pathes = eval.meanLogs(pathes_svgp_random, dsn_svgp_random)
svgp_kmeans_sv_csv_pathes = eval.meanLogs(pathes_svgp_kmeans_sv, dsn_svgp_kmeans_sv)
svgp_kmeans_csv_pathes = eval.meanLogs(pathes_svgp_kmeans, dsn_svgp_kmeans)


plotMetricsForGP([svgp_random_sv_csv_pathes,svgp_random_csv_pathes,svgp_kmeans_sv_csv_pathes,svgp_kmeans_csv_pathes], ["loss","outputscale","likelyhood_noise","test_NLL","test_RMSE", "test_PICP95","test_PICP90","test_PICP50"], 
    max_iteration=150,
    col_labels=["Wine", "Protein", "Parkinsons", "Keggu", "Road"],
    row_labels=["Loss",r"$\sigma^2_f$",r"$\sigma^2_n$", "NLL", "RMSE", "PICP95", "PICP90", "PICP50"],
    group_labels=[r"Random $\sigma^2_f$ T", r"Random $\sigma^2_f$ F",r"K-means $\sigma^2_f$ T", r"K-means $\sigma^2_f$ F"],
    image_filename="assets/svgp_init_comaprison"
)
#--------- CAGP

pathes_cagp_sv, dsn_cagp_sv = eval.getFilePathes(cagp_sv, "results")
pathes_cagp, dsn_cagp= eval.getFilePathes(cagp, "results")

cagp_sv_csv_pathes = eval.meanLogs(pathes_cagp_sv, dsn_cagp_sv)
cagp_csv_pathes = eval.meanLogs(pathes_cagp, dsn_cagp)

plotMetricsForGP([cagp_sv_csv_pathes,cagp_csv_pathes], ["loss","outputscale","likelyhood_noise","test_NLL","test_RMSE", "test_PICP95","test_PICP90","test_PICP50"], 
    max_iteration=150,
    col_labels=["Wine", "Protein", "Parkinsons", "Keggu", "Road"],
    row_labels=["Loss",r"$\sigma^2_f$",r"$\sigma^2_n$", "NLL", "RMSE", "PICP95", "PICP90", "PICP50"],
    group_labels=[r"$\sigma^2_f$ T", r"$\sigma^2_f$ F"],
    image_filename="assets/cagp_sv_comparison"
)

dslist = ["Wine", "Protein", "Parkinson", "Keggu", "Road"]
def getValuesForPathes(pathes, datasets, f):
    for i, path in enumerate(pathes):
        ln, mean_rmse, mean_nll, runtime, std_rmse, std_nll =eval.methodRunData(path)
        f.write(f"{datasets[i]},{ln},{mean_rmse}, {std_rmse}, {mean_nll}, {std_nll}, {runtime}\n")


with open("eval/table.txt", "w") as f:

    f.write("Exact Adam SV\n")
    getValuesForPathes(e_adam_sv_csv_pathes, dslist,f)
    f.write("Exact Adam\n")
    getValuesForPathes(e_adam_csv_pathes, dslist,f)
    f.write("Exact LBFGS SV\n")
    getValuesForPathes(e_lbfgs_sv_csv_pathes, dslist,f)
    f.write("Exact LBFGS\n")
    getValuesForPathes(e_lbfgs_csv_pathes, dslist,f)

    f.write("ExactCG Adam SV\n")
    getValuesForPathes(ecg_adam_sv_csv_pathes, dslist,f)
    f.write("ExactCG Adam\n")
    getValuesForPathes(ecg_adam_csv_pathes, dslist,f)
    f.write("ExactCG LBFGS SV\n")
    getValuesForPathes(ecg_lbfgs_sv_csv_pathes, dslist,f)
    f.write("ExactCG LBFGS\n")
    getValuesForPathes(ecg_lbfgs_csv_pathes, dslist,f)

    f.write("SVGP random SV\n")
    getValuesForPathes(svgp_random_sv_csv_pathes, dslist,f)
    f.write("SVGP random\n")
    getValuesForPathes(svgp_random_csv_pathes, dslist,f)
    f.write("SVGP kmeans SV\n")
    getValuesForPathes(svgp_kmeans_sv_csv_pathes, dslist,f)
    f.write("SVGP kmeans\n")
    getValuesForPathes(svgp_kmeans_csv_pathes, dslist,f)

    f.write("CAGP SV\n")
    getValuesForPathes(cagp_sv_csv_pathes, dslist,f)
    f.write("CAGP\n")
    getValuesForPathes(cagp_csv_pathes, dslist,f)