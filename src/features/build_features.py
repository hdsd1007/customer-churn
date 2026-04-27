import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ── ID-like columns to always drop before encoding ───────────────────────────
_ID_COLS = {"customerid", "customerID", "customer_id", "CustomerID"}

# ── Cardinality ceiling for one-hot encoding ──────────────────────────────────
# Columns with more unique values than this are dropped with a warning
# instead of being one-hot encoded (prevents the OOM bomb)
_MAX_OHE_CARDINALITY = 50


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Deterministic binary encoding for exactly-2-category object columns.
    Operates on the raw series (NOT pre-converted to str) so NaN is preserved.
    """
    # Work with the non-null unique string values only
    vals = sorted(s.dropna().astype(str).unique())

    if set(vals) == {"No", "Yes"}:
        return s.map({"No": 0, "Yes": 1}).astype("int8")

    if set(vals) == {"Female", "Male"}:
        return s.map({"Female": 0, "Male": 1}).astype("int8")

    # Generic: alphabetically lower value → 0, higher → 1
    if len(vals) == 2:
        mapping = {vals[0]: 0, vals[1]: 1}
        return s.astype(str).map(mapping).astype("int8")

    return s  # unchanged if no binary pattern found


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Complete feature-engineering pipeline.

    Parameters
    ----------
    df         : Raw DataFrame (not mutated).
    target_col : Name of the target column; excluded from all encoding.

    Returns
    -------
    ML-ready DataFrame with only numeric columns (plus the target).
    """
    df = df.copy()  # never mutate the caller's DataFrame
    logger.info("Feature engineering started  |  shape: %s", df.shape)

    # ── Step 1: Drop ID columns ───────────────────────────────────────────────
    id_cols_present = [c for c in df.columns if c in _ID_COLS]
    if id_cols_present:
        df = df.drop(columns=id_cols_present)
        logger.info("Dropped ID columns: %s", id_cols_present)

    # ── Step 2: Identify column groups (exclude target) ───────────────────────
    obj_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target_col
    ]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    logger.info(
        "Categorical: %d  |  Numeric: %d", len(obj_cols), len(numeric_cols)
    )

    # ── Step 3: Split object cols into binary vs multi-category ───────────────
    binary_cols = [c for c in obj_cols if df[c].nunique(dropna=True) == 2]
    multi_cols  = [c for c in obj_cols if df[c].nunique(dropna=True) >  2]

    logger.info("Binary cols  : %s", binary_cols)
    logger.info("Multi cols   : %s", multi_cols)

    # ── Step 4: Guard against high-cardinality columns ────────────────────────
    # These would explode memory during get_dummies 
    safe_multi, dropped = [], []
    for c in multi_cols:
        n = df[c].nunique(dropna=True)
        if n > _MAX_OHE_CARDINALITY:
            logger.warning(
                "Dropping '%s': cardinality %d exceeds limit %d",
                c, n, _MAX_OHE_CARDINALITY,
            )
            dropped.append(c)
        else:
            safe_multi.append(c)

    if dropped:
        df = df.drop(columns=dropped)

    # ── Step 5: Binary encoding ───────────────────────────────────────────────
    for c in binary_cols:
        df[c] = _map_binary_series(df[c])   # pass raw series, NOT .astype(str)
        logger.debug("Binary encoded: %s", c)

    # ── Step 6: Convert any bool columns to int8 ──────────────────────────────
    # XGBoost / sklearn expect integers, not Python booleans
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("int8")
        logger.info("Bool → int8: %s", bool_cols)

    # ── Step 7: One-hot encode safe multi-category columns ────────────────────
    if safe_multi:
        before = df.shape[1]
        df = pd.get_dummies(df, columns=safe_multi, drop_first=True, dtype="int8")
        added = df.shape[1] - before
        logger.info(
            "One-hot encoded %d multi-cat cols  |  +%d new features",
            len(safe_multi), added,
        )

    # ── Step 8: Convert remaining OHE bool output to int8 ────────────────────
    # pd.get_dummies with dtype='int8' handles this above, but guard anyway
    leftover_bools = df.select_dtypes(include=["bool"]).columns.tolist()
    if leftover_bools:
        df[leftover_bools] = df[leftover_bools].astype("int8")

    logger.info("Feature engineering complete  |  final shape: %s", df.shape)
    return df