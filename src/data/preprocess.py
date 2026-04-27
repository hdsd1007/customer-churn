import pandas as pd
import logging 

logger = logging.getLogger(__name__)
 
# Columns to force-convert to numeric (may arrive as strings in some exports)
_NUMERIC_COLS = ["TotalCharges", "MonthlyCharges", "ViewingHoursPerWeek", "AverageViewingDuration"]
 
# ID columns to always drop
_ID_COLS = {"customerID", "CustomerID", "customer_id"}
 
 
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning only — no encoding.
    Encoding is handled downstream by build_features.py.
 
    Steps
    -----
    1. Strip whitespace from column names
    2. Drop ID columns
    3. Coerce known numeric columns that may arrive as strings
    """
    pd.options.mode.copy_on_write = True
 
    # Step 1: Tidy headers
    df.columns = df.columns.str.strip()
 
    # Step 2: Drop ID columns
    id_cols_present = [c for c in df.columns if c in _ID_COLS]
    if id_cols_present:
        df = df.drop(columns=id_cols_present)
        logger.info("Dropped ID columns: %s", id_cols_present)
 
    # Step 3: Coerce numeric columns (handles stray strings / whitespace)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    logger.info("Preprocessing complete  |  shape: %s", df.shape)
    return df
    
    
