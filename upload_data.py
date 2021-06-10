import json
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from src.config import COMP_NAME, OUTPUT_PATH


# Remove "=" signs from filenames
def clean_file_names(upload_path):
    for f in upload_path.glob("*/*/*"):
        if "=" in f.stem:
            new_name = f.stem.replace("=", "") + f.suffix
            os.rename(f, f.parent / new_name)
            print(f.stem, "->", new_name)


def create_meta(upload_path, dataset_name):
    meta = {
        "title": dataset_name,
        "id": f"anjum48/{dataset_name}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(upload_path / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)


def upload(checkpoint, dataset_suffix="", dataset_prefix=COMP_NAME):
    upload_path = OUTPUT_PATH / checkpoint
    dataset_name = f"{dataset_prefix}-{checkpoint}"

    if len(dataset_suffix) > 0:
        dataset_name = dataset_name + f"-{dataset_suffix}"

    print("Uploading", upload_path)
    print("Dataset name", dataset_name)

    clean_file_names(upload_path)

    # If new dataset, creata metadata, init and upload
    if not (upload_path / "dataset-metadata.json").is_file():
        create_meta(upload_path, dataset_name)
        create_command = [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(upload_path),
            "-r",
            "zip",
        ]
        subprocess.call(create_command)

    # Otherwise upload new version
    else:
        version_command = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(upload_path),
            "-m",
            "uploading new models",
            "-r",
            "zip",
            "-d",
        ]
        subprocess.call(version_command)


if __name__ == "__main__":
    default_checkpoint = "20210517-143851"

    parser = ArgumentParser()

    parser.add_argument(
        "--timestamp",
        action="store",
        dest="timestamp",
        help="Timestamp for versioning",
        default=default_checkpoint,
        type=str,
    )

    parser.add_argument(
        "--suffix",
        action="store",
        dest="suffix",
        help="Suffix for versioning",
        default="",
        type=str,
    )

    args = parser.parse_args()

    upload(args.timestamp, args.suffix)
