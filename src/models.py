import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AdamW

from src.config import MODEL_CACHE, OUTPUT_PATH
from src.utils import add_weight_decay, get_optimizer_params


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
        kl_loss: bool = False,
        warmup: int = 100,
        hf_config=None,
        pooled: bool = False,
        use_hidden: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if hf_config is None:
            if pretrained:
                model_path = OUTPUT_PATH / "pretraining" / model_name
                print("Using pretrained from", model_path)
                self.config = AutoConfig.from_pretrained(model_name)
                self.transformer = AutoModel.from_pretrained(
                    model_path, output_hidden_states=True
                )
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
            self.config.output_hidden_states = True
            self.transformer = AutoModel.from_config(hf_config)

        # self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        # Multi sample Dropout
        # self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        # self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        # self.regressor = nn.Linear(self.config.hidden_size, 2)
        # self._init_weights(self.layer_norm)
        # self._init_weights(self.regressor)

        if use_hidden:
            n_hidden = self.config.hidden_size * 2
        else:
            n_hidden = self.config.hidden_size

        self.seq_attn_head = nn.Sequential(
            nn.LayerNorm(n_hidden),
            # nn.Dropout(0.1),
            AttentionBlock(n_hidden, n_hidden, 1),
            # nn.Dropout(0.1),
            # nn.Linear(self.config.hidden_size, 2 if kl_loss else 1),
        )

        self.regressor = nn.Linear(n_hidden + 3, 2 if kl_loss else 1)

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
        # out = self.transformer(**kwargs)["logits"]

        model_out = self.transformer(**kwargs)  # 0=seq_output, 1=pooler_output
        # x = self.layer_norm(x)
        # for i, dropout in enumerate(self.dropouts):
        #     if i == 0:
        #         out = self.regressor(dropout(x))
        #     else:
        #         out += self.regressor(dropout(x))
        # out /= len(self.dropouts)

        if self.hparams.use_hidden:
            states = model_out[2]
            out = torch.stack(
                tuple(states[-i - 1] for i in range(self.config.num_hidden_layers)),
                dim=0,
            )
            out_mean = torch.mean(out, dim=0)
            out_max, _ = torch.max(out, dim=0)
            out = torch.cat((out_mean, out_max), dim=-1)
        else:
            out = model_out[0]

        out = self.seq_attn_head(out)
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

        # parameters = get_optimizer_params(self, "a")

        # parameters = [
        #     {
        #         "params": self.transformer.parameters(),
        #         "weight_decay": 0,
        #         "lr": self.hparams.lr,
        #     },
        #     {
        #         "params": [
        #             p for n, p in self.named_parameters() if "transformer" not in n
        #         ],
        #         "weight_decay": self.hparams.weight_decay,
        #         "lr": 1e-3,
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
