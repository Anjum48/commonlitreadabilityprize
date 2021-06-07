from logging import error
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

from src.config import MODEL_CACHE, INPUT_PATH


# https://www.kaggle.com/abhishek/step-1-create-folds
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


class CommonLitDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.excerpt = self.df["excerpt"]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, index):
        row = self.df.loc[index]
        inputs = self.tokenizer(
            str(row["excerpt"]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            # add_special_tokens=True  # not sure what this does
        )

        input_dict = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }

        if "target" in self.df.columns:
            labels = {
                "target": torch.tensor([row["target"]], dtype=torch.float32),
                "error": torch.tensor([row["standard_error"]], dtype=torch.float32),
            }

            # For id 436ce79fe
            if labels["error"] <= 0:
                labels["error"] += 0.5

            labels["target_stoch"] = torch.normal(
                mean=labels["target"], std=labels["error"]
            )
        else:
            labels = 0

        return input_dict, labels


class CommonLitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        model_name: str = "roberta-base",
        max_len: int = 256,
        seed: int = 48,
        folds: int = 5,
        num_workers: int = 16,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE,
        )
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.df = pd.read_csv(INPUT_PATH / "train.csv")
        self.df = create_folds(self.df, folds, seed)

    def setup(self, stage=None, fold_n: int = 0):
        trn_df = self.df.query(f"fold != {fold_n}")
        val_df = self.df.query(f"fold == {fold_n}")

        if stage == "fit" or stage is None:
            self.clr_train = CommonLitDataset(trn_df, self.tokenizer, self.max_len)
            self.clr_valid = CommonLitDataset(val_df, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.clr_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.clr_valid,
            batch_size=128,
            num_workers=self.num_workers,
            pin_memory=True,
        )
