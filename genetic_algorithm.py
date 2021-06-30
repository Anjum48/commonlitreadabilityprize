import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from src.config import INPUT_PATH, OUTPUT_PATH

model_folders = [
    # cherubic-nifty-serval - deberta-large - 0.4836
    "20210614-173633",
    "20210614-203831",
    "20210614-234025",
    "20210615-024138",
    "20210615-054256",
    # scrupulous-mink-of-amplitude - deberta-base - 0.4934
    "20210615-084357",
    "20210615-094729",
    "20210615-105329",
    "20210615-120001",
    "20210615-130640",
    # notorious-sticky-gibbon - roberta-base (with hidden) - 0.4961
    "20210615-220146",
    "20210615-225055",
    "20210615-234038",
    "20210616-003038",
    "20210616-012048",
    # fortunate-cherry-mandrill - roberta-large - 0.4890
    "20210616-021135",
    "20210616-041221",
    "20210616-060255",
    "20210616-075451",
    "20210616-094506",
    # mottled-certain-caracal - distilroberta-base - 0.5076
    "20210616-113626",
    "20210616-121203",
    "20210616-124738",
    "20210616-132341",
    "20210616-140300",
    # aspiring-classic-pegasus - funnel - 0.4975
    "20210617-083847",
    "20210617-102611",
    "20210617-120949",
    "20210617-135233",
    "20210617-153459",
    # silver-bumblebee-of-attack - roberta-base - 0.4932
    "20210617-223340",
    "20210617-232650",
    "20210618-002022",
    "20210618-011405",
    "20210618-020751",
    # sloppy-resourceful-tanuki - albert-large - 0.5241
    "20210617-225903",
    "20210618-010302",
    "20210618-030706",
    "20210618-051049",
    "20210618-071437",
    # rustling-quirky-mastodon - bert-base-uncased - 0.5136
    "20210618-082756",
    "20210618-092115",
    "20210618-100526",
    "20210618-105909",
    "20210618-115253",
    # perky-defiant-husky - bert-large-uncased - 0.5287
    "20210618-124637",
    "20210618-144213",
    "20210618-163942",
    "20210618-183719",
    "20210618-203441",
    # gregarious-brass-perch - bart-base - 0.5445
    "20210618-223208",
    "20210618-233614",
    "20210619-004022",
    "20210619-014809",
    "20210619-025421",
    # military-firefly-of-apotheosis - bart-large - 0.5301
    "20210619-035747",
    "20210619-064351",
    "20210619-093050",
    "20210619-121916",
    "20210619-150740",
    # eccentric-lemur-of-tenacity - sentence-transformers/LaBSE - 0.5230
    "20210622-152356",
    "20210622-161822",
    "20210622-171312",
    "20210622-181238",
    "20210622-191326",
    # valiant-chameleon-of-chaos - sentence-transformers/bert-base-nli-cls-token - 0.5288
    "20210622-165808",
    "20210622-174555",
    "20210622-183427",
    "20210622-192221",
    "20210622-201127",
    # nonchalant-quaint-termite - roberta-base - 0.4951
    "20210623-093223",
    "20210623-101956",
    "20210623-110954",
    "20210623-120004",
    "20210623-125025",
    # skilled-smart-crane - deberta-large (new seeds) - 0.4758
    "20210623-105940",
    "20210623-140343",
    "20210623-170657",
    "20210623-201514",
    "20210623-232231",
    # winged-cerise-agouti - roberta-large - 0.4986
    "20210623-134115",
    "20210623-153240",
    "20210623-172217",
    "20210623-191151",
    "20210623-210342",
    # swift-of-amazing-pride - distilroberta-base - 0.5053
    "20210623-225426",
    "20210623-233019",
    "20210624-000706",
    "20210624-004429",
    "20210624-012102",
    # independent-discerning-earthworm - albert-large-v2
    "20210624-015812",
    "20210624-040309",
    "20210624-060838",
    "20210624-081317",
    "20210624-101855",
    # discreet-visionary-seahorse - microsoft/deberta-base - 0.5181
    "20210624-023057",
    "20210624-033624",
    "20210624-044356",
    "20210624-055212",
    "20210624-070123",
    # chirpy-wren-of-unity - funnel-transformer/large-base - 0.5000
    "20210624-081031",
    "20210624-095223",
    "20210624-113506",
    "20210624-131927",
    "20210624-150250",
    # free-ebony-fennec - microsoft/deberta-base - 0.5021
    "20210627-105133",
    "20210627-115742",
    "20210627-130650",
    "20210627-141604",
    "20210627-152616",
    # blond-viper-of-discussion - deepset/roberta-base-squad2 - 0.4900
    "20210627-105144",
    "20210627-114225",
    "20210627-123605",
    "20210627-133047",
    "20210627-142510",
    # meticulous-demonic-kakapo - roberta-large - 0.5030
    "20210627-151904",
    "20210627-171236",
    "20210627-190737",
    "20210627-210244",
    "20210627-225949",
    # fat-glorious-badger - deepset/roberta-large-squad2 - 0.4937
    "20210628-005835",
    "20210628-025632",
    "20210628-045559",
    "20210628-065437",
    "20210628-085322",
    # beautiful-denim-monkey - funnel-transformer/large-base - 0.5066
    "20210627-163614",
    "20210627-181626",
    "20210627-195827",
    "20210627-213946",
    "20210627-232205",
    # solid-zebu-of-happiness - albert-large-v2 - 0.5207
    "20210628-010737",
    "20210628-031447",
    "20210628-052149",
    "20210628-072849",
    "20210628-093543",
    # parrot-of-strange-maturity - sentence-transformers/LaBSE - 0.5286
    "20210628-114738",
    "20210628-125350",
    "20210628-135845",
    "20210628-150440",
    "20210628-161040",
    # greedy-dog-of-holiness - microsoft/deberta-large - 0.4756
    "20210628-114736",
    "20210628-145921",
    "20210628-181426",
    "20210628-212819",
    "20210629-004241",
    # truthful-hissing-waxbill - deepset/sentence_bert - 0.5504
    "20210628-171705",
    "20210628-180837",
    "20210628-190059",
    "20210628-195246",
    "20210628-204527",
    # blazing-natural-husky - bert-large-cased-whole-word-masking - 0.5202
    "20210628-213743",
    "20210628-233312",
    "20210629-012726",
    "20210629-032224",
    "20210629-051503",
    # ludicrous-heron-of-genius - bert-large-cased
    "20210629-035901",
    "20210629-055338",
    "20210629-074730",
    "20210629-094209",
    "20210629-113421",
    # cooperative-mink-of-spirit - xlm-roberta-base
    "20210629-081350",
    "20210629-091723",
    "20210629-102123",
    "20210629-112540",
    "20210629-122949",
    # dangerous-nebulous-horse - xlm-roberta-large - 0.5092
    "20210629-133352",
    "20210629-154453",
    "20210629-183058",
    "20210629-203803",
    "20210629-224305",
    # passionate-sexy-slug - bart-base - 0.5325
    "20210629-132611",
    "20210629-142628",
    "20210629-152921",
    "20210629-163239",
    "20210629-183052",
]

model_dict = {i: m for i, m in enumerate(model_folders)}


def prep_oofs():
    dataset_paths = [OUTPUT_PATH / f for f in model_folders]

    mpaths, oof_paths = [], []
    for p in dataset_paths:
        mpaths.append(sorted(list(p.glob(f"*/*/*.ckpt"))))
        oof_paths.extend(sorted(list(p.glob(f"*.csv"))))

    oofs = pd.read_csv(
        INPUT_PATH / "train.csv", usecols=["id", "target", "standard_error"]
    ).sort_values(by="id")
    for i, (p, f) in enumerate(zip(oof_paths, model_folders)):
        x = pd.read_csv(p).sort_values(by="id")
        oofs[f] = x["prediction"].values

    return oofs


# https://kaggler.readthedocs.io/en/latest/_modules/kaggler/ensemble/linear.html#netflix
def netflix(es, ps, e0, l=0.0001):
    """Combine predictions with the optimal weights to minimize RMSE.

    Ref: Töscher, A., Jahrer, M., & Bell, R. M. (2009). The bigchaos solution to the netflix grand prize.

    Args:
        es (list of float): RMSEs of predictions
        ps (list of np.array): predictions
        e0 (float): RMSE of all zero prediction
        l (float): lambda as in the ridge regression

    Returns:
        (tuple):

            - (np.array): ensemble predictions
            - (np.array): weights for input predictions
    """
    m = len(es)
    n = len(ps[0])

    X = np.stack(ps).T
    pTy = 0.5 * (n * e0 ** 2 + (X ** 2).sum(axis=0) - n * np.array(es) ** 2)

    w = np.linalg.pinv(X.T.dot(X) + l * n * np.eye(m)).dot(pTy)

    return X.dot(w), w


def get_size(folder):
    # Ubuntu uses 1000**3, Kaggle use 1024**3
    return (
        sum(f.stat().st_size for f in (OUTPUT_PATH / folder).rglob("*") if f.is_file())
        / 1024 ** 3
    )


def get_nf_score(X, y, cv=False):
    if cv:
        scores = []
        for seed in [48, 42, 3]:
            kf = KFold(5, shuffle=True, random_state=seed)

            for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
                train_oofs = X.loc[trn_idx]
                valid_oofs = X.loc[val_idx]
                valid_target = y.loc[val_idx]

                train_preds = [train_oofs[c].values for c in X.columns]
                rmses = [np.sqrt(mean_squared_error(X[c], y)) for c in X.columns]
                _, weights = netflix(rmses, train_preds, 1.4100)

                val_pred = valid_oofs @ weights
                score = np.sqrt(mean_squared_error(val_pred, valid_target))
                scores.append(score)

        return np.mean(scores)
    else:
        preds = [X[c].values for c in X.columns]
        rmses = [np.sqrt(mean_squared_error(X[c], y)) for c in X.columns]
        ensemble, weights = netflix(rmses, preds, 1.4100)
        return np.sqrt(mean_squared_error(ensemble, y))


def obj_func(ints):
    folders = [model_dict[i] for i in ints]
    X = oofs[folders]
    y = oofs["target"]

    # Rename columns due to duplicates
    X.columns = [f"col_{i}" for i in range(len(ints))]

    size = sum([get_size(f) for f in folders])
    score = get_nf_score(X, y)

    if size > 100:
        score += 1

    return score


oofs = prep_oofs()

if __name__ == "__main__":
    varbound = np.array([[0, len(model_folders) - 1]] * 21)

    algorithm_param = {
        "max_num_iteration": 10000,  # Was None
        "population_size": 100,
        "mutation_probability": 0.1,
        "elit_ratio": 0.01,
        "crossover_probability": 0.5,
        "parents_portion": 0.3,
        "crossover_type": "uniform",
        "max_iteration_without_improv": None,
    }

    model = ga(
        function=obj_func,
        dimension=21,
        variable_type="int",
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_param,
    )
    model.run()

    convergence = model.report
    solution = model.output_dict

    print([model_dict[v] for v in solution["variable"]])

    #  [  4.  28.  97.  55.  94.  16.  89. 141. 102. 128.  41. 122.  50.   1.  109.  79.  27.  52.  90. 110.   6.]
