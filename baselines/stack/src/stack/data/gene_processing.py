"""Gene name extraction and filtering utilities."""
from __future__ import annotations

from typing import List, Optional

import h5py
import numpy as np
import pandas as pd


__all__ = [
    "safe_decode_array",
    "filter_gene_names",
    "get_gene_names_from_h5",
]


def safe_decode_array(arr) -> np.ndarray:
    """Decode byte strings from AnnData tables to standard ``str`` values."""
    decoded = []
    for value in arr:
        if isinstance(value, (bytes, bytearray)):
            decoded.append(value.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(value))
    return np.array(decoded, dtype=str)


def filter_gene_names(gene_names: List[str]) -> List[str]:
    """Apply heuristic filters to remove non informative gene identifiers."""
    exclude_patterns = [
        r"^ENS[A-Z]*[0-9]+",
        r"^LOC[0-9]+",
        r"^LINC[0-9]+",
        r"^MT-",
        r"^RP[SL][0-9]",
        r".*P[0-9]*$",
        r"^C[0-9XY]+orf[0-9]+",
        r"^[A-Z]{2}[0-9]{5,}",
        r"\.[0-9]+$",
    ]
    combined_pattern = "|".join(exclude_patterns)

    series = pd.Series(gene_names)
    keep_mask = ~series.str.contains(combined_pattern, regex=True, case=False, na=False)
    filtered = series[keep_mask].tolist()
    return filtered


def get_gene_names_from_h5(
    h5_file_group: h5py.Group,
    gene_name_col: Optional[str] = None,
    use_raw: bool = False,
) -> Optional[np.ndarray]:
    """Return gene names from an H5AD structure, handling categorical encoding."""
    var_group_path = "raw/var" if use_raw and "raw" in h5_file_group else "var"
    if var_group_path not in h5_file_group:
        return None

    var_group = h5_file_group[var_group_path]
    gene_names_source = None

    if gene_name_col and gene_name_col in var_group:
        gene_names_source = var_group[gene_name_col]
    else:
        index_key = "_index" if "_index" in var_group else "index"
        if index_key in var_group:
            gene_names_source = var_group[index_key]

    if gene_names_source is None:
        return None

    gene_names = None
    if isinstance(gene_names_source, h5py.Dataset):
        gene_names = safe_decode_array(gene_names_source[:])
    elif isinstance(gene_names_source, h5py.Group):
        if "categories" in gene_names_source and "codes" in gene_names_source:
            categories = safe_decode_array(gene_names_source["categories"][:])
            codes = gene_names_source["codes"][:]
            valid_mask = codes != -1
            gene_names = np.full(codes.shape, fill_value="", dtype=categories.dtype)
            gene_names[valid_mask] = categories[codes[valid_mask]]

    if gene_names is None:
        return None
    return np.char.upper(gene_names)
