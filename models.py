import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from config import MODEL_CACHE
from utils import add_weight_decay


# https://www.kaggle.com/gogo827jz/roberta-model-parallel-fold-training-on-tpu
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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=MODEL_CACHE
        )
        self.in_features = self.transformer.classifier.dense.in_features
        self.transformer.classifier = nn.Identity()
        # self.att = AttentionBlock(self.in_features, self.in_features, 1)
        self.fc = nn.Linear(self.in_features, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, **kwargs):
        x = self.transformer(**kwargs)["logits"]
        # x = self.att(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_nb):
        inputs, labels = batch
        logits = self(**inputs)
        loss = self.loss_fn(logits, labels["target"])
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("loss/train", avg_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(**inputs)
        loss = self.loss_fn(logits, labels["target"])

        return {
            "val_loss": loss,
            "y_pred": logits,
            "y_true": batch["target"],
            "y_true_sec": batch["target_sec"],
        }

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        y_pred = torch.cat([x["y_pred"] for x in outputs])
        y_true = torch.cat([x["y_true"] for x in outputs])

        self.log_dict(
            {
                "loss/valid": loss_val,
                "rmse": torch.sqrt(loss_val),
            },
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        parameters = add_weight_decay(
            self,
            self.hparams.weight_decay,
            skip_list=["bias", "LayerNorm.bias", "LayerNorm.weight"],
        )

        opt = torch.optim.AdamW(
            parameters,
            lr=self.hparams.lr,
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return [opt], [sch]
