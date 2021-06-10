import getpass
from pathlib import Path

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, RidgeCV
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AdamW,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

KERNEL = False if getpass.getuser() == "anjum" else True

if not KERNEL:
    INPUT_PATH = Path("/mnt/storage_dimm2/kaggle_data/commonlitreadabilityprize")
    OUTPUT_PATH = Path("/mnt/storage/kaggle_output/commonlitreadabilityprize")
    MODEL_CACHE = Path("/mnt/storage/model_cache/torch")
else:
    INPUT_PATH = Path("../input/commonlitreadabilityprize")
    MODEL_CACHE = None

    # Install packages
    import subprocess

    whls = [
        "../input/textstat/Pyphen-0.10.0-py3-none-any.whl",
        "../input/textstat/textstat-0.7.0-py3-none-any.whl",
    ]

    for w in whls:
        subprocess.call(["pip", "install", w, "--no-deps"])

import textstat


# models.py
class AttentionBlock(nn.Module):
    def __init__(self, in_features, middle_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.middle_features = middle_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, middle_features)
        self.V = nn.Linear(middle_features, out_features)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


class CommonLitModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "roberta-base",
        lr: float = 0.001,
        weight_decay: float = 0,
        pretrained: bool = False,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        kl_loss: bool = False,
        warmup: int = 100,
        hf_config=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if hf_config is None:
            if pretrained:
                model_path = OUTPUT_PATH / "pretraining" / model_name
                print("Using pretrained from", model_path)
                self.config = AutoConfig.from_pretrained(model_name)
                self.transformer = AutoModel.from_pretrained(model_path)
            else:
                self.config = AutoConfig.from_pretrained(
                    model_name,
                    cache_dir=MODEL_CACHE / model_name,
                )
                self.transformer = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=MODEL_CACHE / model_name,
                    output_hidden_states=True,
                )
        else:
            self.config = hf_config
            self.transformer = AutoModel.from_config(hf_config)

        self.seq_attn_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            # nn.Dropout(0.1),
            AttentionBlock(self.config.hidden_size, self.config.hidden_size, 1),
            # nn.Dropout(0.1),
            # nn.Linear(self.config.hidden_size, 2 if kl_loss else 1),
        )

        self.regressor = nn.Linear(self.config.hidden_size + 3, 2 if kl_loss else 1)
        self.loss_fn = nn.MSELoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, **kwargs):
        x = self.transformer(**kwargs)[0]  # 0=seq_output, 1=pooler_output

        out = self.seq_attn_head(x)
        out = torch.cat([out, features], -1)
        out = self.regressor(out)

        if out.shape[1] == 1:
            return out, None
        else:
            mean = out[:, 0].view(-1, 1)
            log_var = out[:, 1].view(-1, 1)
            return mean, log_var

    def training_step(self, batch, batch_nb):
        inputs, labels, features = batch
        mean, log_var = self.forward(features, **inputs)
        if self.hparams.kl_loss:
            p = torch.distributions.Normal(mean, torch.exp(log_var))
            q = torch.distributions.Normal(labels["target"], labels["error"])
            loss = torch.distributions.kl_divergence(p, q).mean()
        else:
            loss = self.loss_fn(mean, labels["target"])
        self.log_dict({"loss/train_step": loss})
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("loss/train", avg_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inputs, labels, features = batch
        mean, log_var = self.forward(features, **inputs)
        if self.hparams.kl_loss:
            p = torch.distributions.Normal(mean, torch.exp(log_var))
            q = torch.distributions.Normal(labels["target"], labels["error"])
            loss = torch.distributions.kl_divergence(p, q).mean()
        else:
            loss = self.loss_fn(mean, labels["target"])

        return {
            "val_loss": loss,
            "y_pred": mean,
            "y_true": labels["target"],
        }

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        y_pred = torch.cat([x["y_pred"] for x in outputs])
        y_true = torch.cat([x["y_true"] for x in outputs])

        rmse = torch.sqrt(self.loss_fn(y_pred, y_true))

        self.log_dict(
            {
                "loss/valid": loss_val,
                "rmse": rmse,
            },
            prog_bar=True,
            sync_dist=True,
        )

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # Warm-up the first 100 steps
        if self.trainer.global_step < self.hparams.warmup:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        parameters = add_weight_decay(
            self,
            self.hparams.weight_decay,
            skip_list=["bias", "LayerNorm.bias", "LayerNorm.weight"],
        )

        opt = AdamW(
            parameters,
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return [opt], [sch]


# utils.py
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or any(s in name for s in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


# datasets.py
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

        # Add addtional features
        features = self.generate_features(str(row["excerpt"]))

        return input_dict, labels, features

    def generate_features(self, text):
        means = torch.tensor([8.564220, 172.948483, 67.742121])
        stds = torch.tensor([3.666797, 16.974894, 17.530230])
        features = torch.tensor(
            [
                textstat.sentence_count(text),
                textstat.lexicon_count(text),
                textstat.flesch_reading_ease(text),
            ]
        )
        return (features - means) / stds


# infer.py
def infer(model, dataset, batch_size=128, device="cuda"):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    predictions = []
    with torch.no_grad():
        for input_dict, _, features in loader:
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            mean, log_var = model(features.to(device), **input_dict)
            predictions.append(mean.cpu())

    return torch.cat(predictions, 0)


def make_predictions(dataset_paths, device="cuda"):
    mpaths, oof_paths = [], []
    for p in dataset_paths:
        mpaths.append(sorted(list(p.glob(f"*/*/*.ckpt"))))
        oof_paths.extend(sorted(list(p.glob(f"*.csv"))))

    print(
        f"{len([item for sublist in mpaths for item in sublist])} models found.",
        f"{len(oof_paths)} OOFs found",
    )

    # Construct OOF df
    oofs = pd.read_csv(INPUT_PATH / "train.csv", usecols=["id", "target"]).sort_values(
        by="id"
    )
    for i, p in enumerate(oof_paths):
        x = pd.read_csv(p).sort_values(by="id")
        oofs[f"model_{i}"] = x["prediction"].values

    df = pd.read_csv(INPUT_PATH / "test.csv")
    output = 0

    for i, group in enumerate(mpaths):
        output = 0
        for p in group:
            print(p)
            config = AutoConfig.from_pretrained(str(p.parent))
            tokenizer = AutoTokenizer.from_pretrained(str(p.parent))
            model = CommonLitModel.load_from_checkpoint(p, hf_config=config)
            dataset = CommonLitDataset(df, tokenizer)
            output += infer(model, dataset, device=device)

        df[f"model_{i}"] = output.squeeze().numpy() / len(group)

    pred_cols = [f"model_{i}" for i in range(len(mpaths))]

    # Use mean
    # df["target"] = df[pred_cols].mean(1)

    # Stack using linear regression
    print(oofs[pred_cols].corr())
    reg = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50, 100, 500, 1000))
    reg.fit(oofs[pred_cols], oofs["target"])
    print(f"Weights: {reg.coef_}, bias: {reg.intercept_}")
    print(f"Best RMSE: {np.sqrt(-reg.best_score_):0.5f}. Alpha {reg.alpha_}")
    df["target"] = reg.predict(df[pred_cols])

    df[["id", "target"]].to_csv("submission.csv", index=False)


if __name__ == "__main__":

    model_folders = [
        # proud-flashy-yak
        # "20210608-183327",
        # "20210608-190544",
        # "20210608-193801",
        # complex-heron-of-science
        "20210609-171109",
        "20210609-174639",
        "20210609-182121",
        "20210609-192843",
        "20210609-200242",
        # impetuous-marvellous-cockle
        "20210608-233655",
        "20210609-004922",
        "20210609-020213",
        "20210609-205046",
        "20210609-220344",
        # zippy-caped-leech
        "20210609-125306",
        "20210609-141352",
        # "20210609-154233",
    ]

    if KERNEL:
        dataset_paths = [
            Path(f"../input/commonlitreadabilityprize-{f}") for f in model_folders
        ]
    else:
        dataset_paths = [OUTPUT_PATH / f for f in model_folders]

    predictions = make_predictions(dataset_paths, device="cuda:1")
