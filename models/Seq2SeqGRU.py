import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from torch import nn, optim
from torchmetrics import Accuracy, F1Score
from yaml.loader import SafeLoader

from CL.common import types

CONFIG = yaml.load(
    (Path(os.environ["PROJECT_PATH"]) / "config.yaml").open(),
    Loader=SafeLoader,
)
MODEL_CONFIG = CONFIG["models"]["seq2seqGRU"]
CL_CONFIG = CONFIG["curriculum"]


class Seq2SeqGRU(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers=1, lr=1e-3
    ):
        super(Seq2SeqGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lr = lr
        self.loss_func = nn.CrossEntropyLoss()

        self.emb = nn.Embedding(
            num_embeddings=input_size, embedding_dim=hidden_size
        )
        self.dec_emb = nn.Embedding(
            num_embeddings=output_size, embedding_dim=hidden_size
        )

        # Encoder
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.acc_metric = Accuracy(
            task="multiclass", num_classes=self.output_size
        )
        self.f1_metric = F1Score(
            task="multiclass", num_classes=self.output_size
        )

    def forward(self, x, y=None, use_tf=False):
        BOS = types.SpecialTokens.BOS.value
        batch_size = y.size(0)
        sent_length = y.size(1)
        y_pred = []

        encoder_input = self.emb(x)

        decoder_input = torch.tensor(
            [[BOS] * batch_size], dtype=torch.long, device=self.device
        ).view(batch_size, 1)

        # Encoder
        _, hidden = self.encoder(encoder_input)

        # Decoder
        for i in range(sent_length):
            decoder_input = self.dec_emb(decoder_input)

            (
                decoder_output,
                hidden,
            ) = self.decoder(decoder_input, hidden)
            outputs = self.fc(decoder_output)

            if use_tf:
                decoder_input = y[:, i].view(batch_size, 1)
            else:
                _, top_index = outputs.topk(1)
                decoder_input = top_index.view(batch_size, 1)

            y_pred.append(outputs[:, 0, :])
        model_output = torch.stack(y_pred, dim=1)
        return model_output

    def training_step(self, batch, batch_idx):
        x, y = batch
        use_tf = np.random.binomial(1, 0.8)

        y_pred = self(x, y, use_tf)
        loss = self.loss_func(
            y_pred.view(-1, self.output_size),
            y.view(-1),
        )

        acc_score = self.acc_metric(
            y_pred.reshape(-1, self.output_size), y.reshape(-1)
        )
        f1_score = self.f1_metric(
            y_pred.reshape(-1, self.output_size), y.reshape(-1)
        )
        self.log("train_exact_match", acc_score, prog_bar=True, logger=True)
        self.log("train_f1_score", f1_score, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        use_tf = False

        y_pred = self(x, y, use_tf)
        loss = self.loss_func(
            y_pred.view(-1, self.output_size),
            y.view(-1),
        )

        acc_score = self.acc_metric(
            y_pred.reshape(-1, self.output_size), y.reshape(-1)
        )
        f1_score = self.f1_metric(
            y_pred.reshape(-1, self.output_size),
            y.reshape(-1),
        )
        self.log("val_exact_match", acc_score, prog_bar=True, logger=True)
        self.log("val_f1_score", f1_score, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
