import re
import pandas as pd
import gspread
import yaml
from gspread_dataframe import set_with_dataframe
from config import OUTPUT_PATH


def aggregate_scores(n_folders=0):
    folders = [p for p in OUTPUT_PATH.iterdir() if p.stem[-1].isnumeric()]
    folders.sort()
    agg_scores = []

    for f in folders[-n_folders:]:
        scores = []
        for ckpt in sorted(f.glob("*/*/*.ckpt")):

            # Get the slug & seed
            if len(scores) == 0:
                with open(ckpt.parent / "hparams.yaml", "r") as ymlfile:
                    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
                    scores.append(cfg["slug"])
                    scores.append(f.name)
                    scores.append(cfg["seed"])

            # epoch=02-rmse=0.5375 - match the last number. = sign optional
            scores.append(float(re.findall(r"rmse=?(0\.\d+)", ckpt.stem)[0]))
        agg_scores.append(scores)

    return agg_scores


def write_to_gspread(scores):
    df = pd.DataFrame(scores)
    df.columns = [
        "Group name",
        "Timestamp",
        "Seed",
        "Fold 0",
        "Fold 1",
        "Fold 2",
        "Fold 3",
        "Fold 4",
    ]

    gc = gspread.service_account()
    sh = gc.open("commonlit_readability")
    worksheet = sh.get_worksheet(0)

    # empty_row = len(worksheet.col_values(1)) + 1
    # worksheet.update(f"A{empty_row}", df.values.tolist())

    # Refresh the full df
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())


if __name__ == "__main__":
    scores = aggregate_scores()
    write_to_gspread(scores)
