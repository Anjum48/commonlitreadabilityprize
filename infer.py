import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.config import INPUT_PATH, OUTPUT_PATH
from src.datasets import CommonLitDataset, create_folds
from src.models import CommonLitModel


def infer(model, dataset, batch_size=64, device="cuda"):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    predictions = []
    with torch.no_grad():
        for input_dict, _, features in loader:
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            mean, log_var = model(features.to(device), **input_dict)
            predictions.append(mean.cpu())

    return torch.cat(predictions, 0)


def make_oofs(folder_name, seed, device="cuda"):
    mpaths = sorted(list((OUTPUT_PATH / folder_name).glob(f"*/*/*.ckpt")))
    tokenizers = [AutoTokenizer.from_pretrained(str(p.parent)) for p in mpaths]
    configs = [AutoConfig.from_pretrained(str(p.parent)) for p in mpaths]
    models = [
        CommonLitModel.load_from_checkpoint(p, hf_config=c)
        for p, c in zip(mpaths, configs)
    ]
    print(
        f"{len(mpaths)} models found.",
        f"{len(tokenizers)} tokenizers found.",
        f"{len(configs)} configs found",
    )

    df = pd.read_csv(INPUT_PATH / "train.csv")
    df = create_folds(df, 5, seed)
    df["prediction"] = 0

    for fold, (model, tokenizer) in enumerate(zip(models, tokenizers)):
        df_fold = df.query(f"fold == {fold}")
        dataset = CommonLitDataset(df_fold, tokenizer)
        df.loc[df_fold.index, "prediction"] = (
            infer(model, dataset, device=device).squeeze().numpy()
        )

    rmse = np.sqrt(mean_squared_error(df["prediction"], df["target"]))
    print(f"OOF RMSE {rmse:0.5f}")
    df.to_csv(OUTPUT_PATH / folder_name / f"oofs_{rmse:0.5f}.csv", index=False)


if __name__ == "__main__":

    default_checkpoint = "20210607-205257"

    parser = ArgumentParser()

    parser.add_argument(
        "--timestamp",
        action="store",
        dest="timestamp",
        help="Timestamp for versioning",
        default=default_checkpoint,
        type=str,
    )

    parser.add_argument(
        "--seed",
        action="store",
        dest="seed",
        help="Seed used for splits",
        default=48,
        type=int,
    )

    parser.add_argument(
        "--gpu",
        action="store",
        dest="gpu",
        help="GPU index to use",
        default="0",
        type=str,
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    predictions = make_oofs(args.timestamp, args.seed, device="cuda")
