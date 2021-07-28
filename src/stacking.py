import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from config import INPUT_PATH, OUTPUT_PATH

# from src.datasets import create_folds
from model_folders import model_folders


def create_folds(data, n_splits, random_state=None):
    # we create a new column called fold and fill it with -1
    data["fold"] = -1

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin targets
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)

    # initiate the kfold class from model_selection module
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, "fold"] = f

    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data


def build_oof_df(folders):
    dataset_paths = [OUTPUT_PATH / f for f in folders]
    mpaths, oof_paths = [], []
    for p in dataset_paths:
        mpaths.append(sorted(list(p.glob(f"*/*/*.ckpt"))))
        oof_paths.extend(sorted(list(p.glob(f"*.csv"))))

    oofs = pd.read_csv(
        INPUT_PATH / "train.csv", usecols=["id", "target", "standard_error"]
    ).sort_values(by="id")
    for i, p in enumerate(oof_paths):
        x = pd.read_csv(p).sort_values(by="id")
        oofs[p.parent.name] = x["prediction"].values

    return oofs.reset_index(drop=True)


def scorer_lstsq(oofs, folders, device="cuda:1"):
    data = oofs[folders].values
    target = oofs["target"].values.reshape(-1, 1)

    loo = LeaveOneOut()

    train_X, train_y, valid_X, valid_y = [], [], [], []
    for trn_idx, val_idx in loo.split(data):
        train_X.append(data[trn_idx])
        train_y.append(target[trn_idx])
        valid_X.append(data[val_idx])
        valid_y.append(target[val_idx])

    train_X = torch.tensor(np.stack(train_X), dtype=torch.float32).to(device)
    train_y = torch.tensor(np.stack(train_y), dtype=torch.float32).to(device)
    valid_X = torch.tensor(np.stack(valid_X), dtype=torch.float32).to(device)
    valid_y = torch.tensor(np.stack(valid_y), dtype=torch.float32).to(device)

    # Same as (np.linalg.pinv(X.T @ X) @ X.T) @ y
    W = torch.linalg.lstsq(train_X, train_y).solution
    y_pred = valid_X @ W
    mse = torch.nn.functional.mse_loss(y_pred, valid_y)
    return torch.sqrt(mse).cpu().numpy()


def scorer_bayesian_ridge(oofs, folders, folds=10):
    oofs = create_folds(oofs, folds, 48)
    fold_scores = []

    for fold in range(folds):
        trn_df = oofs.query(f"fold != {fold}")
        val_df = oofs.query(f"fold == {fold}")

        reg = BayesianRidge(tol=1e-4, fit_intercept=False)
        reg.fit(trn_df[folders], trn_df["target"])
        y_pred = reg.predict(val_df[folders])

        mse = mean_squared_error(y_pred, val_df["target"])
        fold_scores.append(mse)

    return np.sqrt(np.mean(fold_scores))


def scorer_ridge(oofs, folders):
    reg = RidgeCV(
        alphas=(
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            5.0,
            10.0,
            50,
            100,
            500,
            1000,
        ),
        normalize=True,
    )

    reg.fit(oofs[folders], oofs["target"])
    return np.sqrt(-reg.best_score_)


def get_size(folder):
    # Ubuntu uses 1000**3, Kaggle use 1024**3
    return (
        sum(f.stat().st_size for f in (OUTPUT_PATH / folder).rglob("*") if f.is_file())
        / 1024 ** 3
    )


def pruning(oofs, scorer=scorer_lstsq, candidates=model_folders):
    history = []
    score = scorer(oofs, candidates)
    print(f"Initial score {score:0.5f}")

    while len(candidates) > 1:
        trial_candidates = [
            candidates[:i] + candidates[i + 1 :] for i, _ in enumerate(candidates)
        ]
        scores = [scorer(oofs, tc) for tc in trial_candidates]
        removed = candidates[np.argmin(scores)]
        del candidates[np.argmin(scores)]
        score = scorer(oofs, candidates)
        size = np.sum([get_size(c) for c in candidates])
        history.append(
            {
                "models": candidates.copy(),
                "score": score,
                "size": size,
                "removed": removed,
            }
        )
        print(
            f"{len(history)} New score {score:0.5f}. Size: {size:0.1f} GB.",
            f"Removed {removed}. {len(candidates)} models",
        )

    history = pd.DataFrame(history)
    history.to_csv("pruning_bayesian.csv", index=False)
    print(history.tail(40))


if __name__ == "__main__":
    oofs = build_oof_df(model_folders)
    pruning(oofs, scorer_bayesian_ridge, model_folders)
    # pruning(oofs, scorer_ridge, model_folders)
