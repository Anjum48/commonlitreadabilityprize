import torch
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import LeaveOneOut
from config import INPUT_PATH, OUTPUT_PATH
from model_folders import model_folders


def build_oof_df():
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


def scorer(oofs, folders, device="cuda:1"):
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


def get_size(folder):
    # Ubuntu uses 1000**3, Kaggle use 1024**3
    return (
        sum(f.stat().st_size for f in (OUTPUT_PATH / folder).rglob("*") if f.is_file())
        / 1024 ** 3
    )


def pruning(oofs):
    candidates = model_folders.copy()
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
            f"{len(history)} New score {score:0.5f}. Size: {size:0.1f} GB. Removed {removed}"
        )

    history = pd.DataFrame(history)
    history.to_csv("pruning_lstsq.csv", index=False)
    print(history.tail(40))


if __name__ == "__main__":
    oofs = build_oof_df()
    pruning(oofs)
