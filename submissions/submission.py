import getpass
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
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
    # import subprocess

    # whls = [
    #     "../input/timm-pytorch-image-models/pytorch-image-models-master",
    #     "../input/torchlibrosa/torchlibrosa-0.0.9-py2.py3-none-any.whl",
    # ]

    # for w in whls:
    #     subprocess.call(["pip", "install", w, "--no-deps"])


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
        config=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if config is None:
            self.config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE,
                output_hidden_states=True,
                local_files_only=True,
            )
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)

        self.seq_attn_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            # nn.Dropout(0.1),
            AttentionBlock(self.config.hidden_size, self.config.hidden_size, 1),
            # nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, 1),
        )

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

    def forward(self, **kwargs):
        x = self.transformer(**kwargs)[0]  # 0=seq_output, 1=pooler_output
        out = self.seq_attn_head(x)

        if out.shape[1] == 1:
            return out, None
        else:
            mean = out[:, 0].view(-1, 1)
            log_var = out[:, 1].view(-1, 1)
            return mean, log_var

    def training_step(self, batch, batch_nb):
        inputs, labels = batch
        mean, log_var = self.forward(**inputs)
        # p = torch.distributions.Normal(mean, torch.exp(log_var))
        # q = torch.distributions.Normal(labels["target"], labels["error"])
        # loss = torch.distributions.kl_divergence(p, q).mean()
        loss = self.loss_fn(mean, labels["target"])
        self.log_dict({"loss/train_step": loss})
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("loss/train", avg_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        mean, log_var = self.forward(**inputs)
        # p = torch.distributions.Normal(mean, torch.exp(log_var))
        # q = torch.distributions.Normal(labels["target"], labels["error"])
        # loss = torch.distributions.kl_divergence(p, q).mean()
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
            {"loss/valid": loss_val, "rmse": rmse},
            prog_bar=True,
            sync_dist=True,
        )

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

        return input_dict, labels


# infer.py
def infer(model, dataset, batch_size=128, device="cuda"):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    predictions = []
    with torch.no_grad():
        for input_dict, _ in loader:
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            mean, log_var = model(**input_dict)
            predictions.append(mean.cpu())

    return torch.cat(predictions, 0)


def make_predictions(dataset_paths):
    mpaths = []
    for p in dataset_paths:
        mpaths.extend(list(p.glob(f"*/*.ckpt")))
    mpaths.sort()
    tokenizers = [AutoTokenizer.from_pretrained(str(p.parent)) for p in mpaths]
    configs = [AutoConfig.from_pretrained(str(p.parent)) for p in mpaths]
    models = [
        CommonLitModel.load_from_checkpoint(p, config=c)
        for p, c in zip(mpaths, configs)
    ]
    print(
        f"{len(mpaths)} models found.",
        f"{len(tokenizers)} tokenizers found.",
        f"{len(configs)} configs found",
    )

    df = pd.read_csv(INPUT_PATH / "test.csv")

    for model, tokenizer in zip(models, tokenizers):
        dataset = CommonLitDataset(df, tokenizer)
        output = infer(model, dataset)

    output /= len(models)

    df["target"] = output.squeeze().numpy()
    df[["id", "target"]].to_csv("submission.csv", index=False)


if __name__ == "__main__":

    dataset_paths = [
        OUTPUT_PATH / "20210605-150515" / "roberta-base-squad2",
        # OUTPUT_PATH / "20210605-153747" / "roberta-base-squad2",
        # OUTPUT_PATH / "20210605-160907" / "roberta-base-squad2",
    ]

    predictions = make_predictions(dataset_paths)
