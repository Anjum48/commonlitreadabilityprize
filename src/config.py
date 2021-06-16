from pathlib import Path

COMP_NAME = "commonlitreadabilityprize"

INPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_data/{COMP_NAME}/")
OUTPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_output/{COMP_NAME}/")
CONFIG_PATH = Path(f"/home/anjum/kaggle/{COMP_NAME}/hyperparams.yml")
MODEL_CACHE = Path("/mnt/storage/model_cache/huggingface")
