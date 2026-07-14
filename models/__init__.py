"""models/ — baseline LSTM identity classifier for the BFId lab.

Implements the deliberately *simple* pipeline described by Todt, Morsbach,
Strufe ("BFId: Identity Inference Attacks Utilizing Beamforming Feedback
Information", CCS '25, sec. 5.2 "Classification"):

    feature standardisation ONLY  (no handcrafted signal preprocessing)
        -> LSTM (variable-length sequences via pack_padded_sequence)
        -> FC -> BatchNorm1d -> ReLU -> FC
        -> softmax classifier (CrossEntropyLoss)
        Adam optimiser, default 80/20 train/test split.

The headline paper result is BFI identity accuracy 99.5% +- 0.38 from normal
walking; this module reproduces the *pipeline*, not the dataset.

Modules
-------
* :mod:`models.dataset`          torch Dataset over ``samples.parquet`` + npz.
* :mod:`models.lstm_classifier`  the :class:`BFIdLSTM` nn.Module.
* :mod:`models.train`            training CLI (yaml/flags) -> data/models/.
* :mod:`models.evaluate`         held-out evaluation w/ paper breakdowns.
* :mod:`models.metrics`          accuracy / top-2 / per-group / confusion.

All modules import dims/paths/labels from :mod:`wallflower.contract` so naming and
feature dimensions never drift. The heavy stack (numpy, torch, pandas, pyarrow,
scikit-learn) lives in the ``[ml]`` optional dependency group.
"""
from __future__ import annotations

__all__ = [
    "dataset",
    "lstm_classifier",
    "train",
    "evaluate",
    "metrics",
]
