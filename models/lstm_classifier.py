"""models.lstm_classifier — the baseline :class:`BFIdLSTM` network.

Reproduces the deliberately simple classifier of Todt, Morsbach, Strufe
(CCS '25, sec. 5.2): a (packed) LSTM consumes the variable-length, standardised
feature sequence; the final hidden state feeds two fully-connected layers with
BatchNorm + ReLU between them, ending in a linear layer whose logits are turned
into a probability distribution by :class:`torch.nn.CrossEntropyLoss`
(softmax + NLL) during training.

    x: [B, T_max, input_dim]  (padded)        lengths: [B]
        -> pack_padded_sequence
        -> LSTM (hidden, lstm_layers)
        -> last layer's final hidden state h_n[-1]  [B, hidden]
        -> Linear(hidden -> fc_hidden)
        -> BatchNorm1d(fc_hidden) -> ReLU
        -> Linear(fc_hidden -> num_classes)        => logits

``input_dim`` must equal ``contract.expected_feature_dim(modality, with_dt=True)``
because :mod:`models.dataset` appends the per-step time-delta as the final
feature column.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BFIdLSTM(nn.Module):
    """Baseline BFId identity classifier (paper sec. 5.2).

    Parameters
    ----------
    input_dim
        Per-timestep feature width incl. the appended dt column, i.e.
        ``contract.expected_feature_dim(modality, with_dt=True)``.
    num_classes
        Number of distinct participant ids in the label vocabulary.
    hidden
        LSTM hidden size (paper tuned per-modality via Optuna).
    lstm_layers
        Number of stacked LSTM layers.
    fc_hidden
        Width of the intermediate fully-connected layer.
    dropout
        Dropout applied between LSTM layers (only effective if lstm_layers > 1)
        and before the final FC layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 128,
        lstm_layers: int = 2,
        fc_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if num_classes < 2:
            raise ValueError("need at least 2 classes for a classifier")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.lstm_layers = int(lstm_layers)
        self.fc_hidden = int(fc_hidden)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=float(dropout) if self.lstm_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(self.hidden, self.fc_hidden)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(self.fc_hidden, self.num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Map a padded batch to class logits.

        Parameters
        ----------
        x : ``[B, T_max, input_dim]`` padded float tensor.
        lengths : ``[B]`` int tensor of true (unpadded) sequence lengths.

        Returns ``[B, num_classes]`` logits (softmax handled by the loss).
        """
        # pack_padded_sequence needs lengths on CPU as int64 (enforce + sort).
        lengths_cpu = lengths.detach().to("cpu").long()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        # h_n: [num_layers, B, hidden]; take the last layer's final hidden state.
        _, (h_n, _) = self.lstm(packed)
        last = h_n[-1]                       # [B, hidden]

        z = self.fc1(last)                   # [B, fc_hidden]
        z = self.bn1(z)
        z = self.relu(z)
        z = self.drop(z)
        logits = self.fc2(z)                 # [B, num_classes]
        return logits

    # -- (de)serialisation helpers so train/evaluate stay in sync ----------- #
    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden": self.hidden,
            "lstm_layers": self.lstm_layers,
            "fc_hidden": self.fc_hidden,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "BFIdLSTM":
        return cls(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
            hidden=cfg.get("hidden", 128),
            lstm_layers=cfg.get("lstm_layers", 2),
            fc_hidden=cfg.get("fc_hidden", 64),
        )
