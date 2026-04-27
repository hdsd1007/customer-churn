import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")   # Fix: was FileExistsError
    df = pd.read_csv(file_path)
    logger.info("Loaded %s  |  shape: %s", file_path, df.shape)
    return df