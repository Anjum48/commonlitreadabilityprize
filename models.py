import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModel,
    AdamW,
)

from config import MODEL_CACHE, OUTPUT_PATH
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
        pretrained: bool = False,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if pretrained:
            model_path = OUTPUT_PATH / "pretraining" / model_name
            self.transformer = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=1
            )
        else:
            self.transformer = AutoModelForSequenceClassification.from_pretrained(
                model_name, cache_dir=MODEL_CACHE, num_labels=1
            )

        # https://www.kaggle.com/rhtsingh/two-roberta-s-are-better-than-one-0-469
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, cache_dir=MODEL_CACHE)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        # Multi sample DO
        # self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        self.regressor = nn.Linear(self.config.hidden_size, 2)
        self._init_weights(self.layer_norm)
        self._init_weights(self.regressor)

        # self.transformer = AutoModelForSequenceClassification.from_pretrained(
        #     model_name, cache_dir=MODEL_CACHE
        # )
        # self.in_features = self.transformer.classifier.dense.in_features
        # self.transformer.classifier.out_proj = nn.Linear(self.in_features, 2)
        # self.transformer.classifier = nn.Identity()
        # self.att = AttentionBlock(self.in_features, self.in_features, 1)
        # self.fc = nn.Linear(self.in_features, 2)
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
        # out = self.transformer(**kwargs)["logits"]

        x = self.transformer(**kwargs)[1]  # pooler_output
        x = self.layer_norm(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.regressor(dropout(x))
            else:
                out += self.regressor(dropout(x))
        out /= len(self.dropouts)

        # x = self.att(x)
        # x = self.fc(x)
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
            {
                "loss/valid": loss_val,
                "rmse": rmse,
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

        # parameters = [
        #     {
        #         "params": self.transformer.roberta.parameters(),
        #         "weight_decay": 0,  # self.hparams.weight_decay,
        #         "lr": self.hparams.lr,
        #     },
        #     {
        #         "params": self.transformer.classifier.parameters(),
        #         "weight_decay": self.hparams.weight_decay,
        #         "lr": 0.001,
        #     },
        # ]

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
