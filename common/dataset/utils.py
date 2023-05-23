import json
import os
from pathlib import Path


def load(config: dict) -> tuple[list[dict], list[dict]]:
    # TODO: add validation on config fields
    train_data = json.load(
        (
            Path(os.environ["PROJECT_PATH"])
            / Path(config["root_path"])
            / Path(config["train"]["basket_name"])
        ).open()
    )

    test_data = json.load(
        (
            Path(os.environ["PROJECT_PATH"])
            / Path(config["root_path"])
            / Path(config["test"]["basket_name"])
        ).open()
    )

    return train_data, test_data


if __name__ == "__main__":
    train, test = load()
