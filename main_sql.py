import os
from pathlib import Path

import yaml

# from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from yaml.loader import SafeLoader

from CL.common.dataset.pauq.make_dataset import Text2SQLDataModule
from CL.curriculum.callback import CLScheduler
from CL.models.Seq2SeqGRU import Seq2SeqGRU

# logger = pl_loggers.WandbLogger(save_dir="lightning_logs/")

seed_everything(42, workers=True)

CONFIG = yaml.load(
    (Path(os.environ["PROJECT_PATH"]) / "config.yaml").open(),
    Loader=SafeLoader,
)
PAUQ_CONFIG = CONFIG["dataset"]["pauq"]
MODEL_CONFIG = CONFIG["models"]["seq2seqGRU"]
CL_CONFIG = CONFIG["curriculum"]


def main():
    datamodule = Text2SQLDataModule(
        batch_size=MODEL_CONFIG["batch_size"],
    )
    datamodule.setup()
    enc_vocab_size = len(datamodule.src_tokenizer)
    dec_vocab_size = len(datamodule.trg_tokenizer)
    # model
    model = Seq2SeqGRU(
        input_size=enc_vocab_size,
        hidden_size=MODEL_CONFIG["hidden_size"],
        output_size=dec_vocab_size,
        num_layers=MODEL_CONFIG["num_layers"],
        lr=MODEL_CONFIG["lr"],
    )
    trainer_args = {
        "accelerator": "cuda" if MODEL_CONFIG["device"] == "cuda" else "cpu",
        "max_epochs": MODEL_CONFIG["num_epoches"],
        "gradient_clip_val": MODEL_CONFIG["grad_clip"],
        "enable_progress_bar": True,
        "reload_dataloaders_every_n_epochs": 1,
        "deterministic": True,
        "callbacks": [
            EarlyStopping(
                monitor="val_exact_match",
                min_delta=0.01,
                patience=10,
                verbose=False,
                mode="max",
            ),
            CLScheduler(
                n_steps=CL_CONFIG["n_steps"],
                min_fraction=CL_CONFIG["min_fraction"],
            ),
        ],
        # "logger": logger,
        # "default_root_dir": (
        #     "/Users/darby/Desktop/MIPT/ML/CL/CL/lightning_logs"
        # ),
        "log_every_n_steps": 4,
    }

    # training
    trainer = Trainer(**trainer_args)
    trainer.fit(
        model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
