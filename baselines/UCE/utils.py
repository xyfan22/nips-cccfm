"""
Utils

"""

import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
import numpy as np
import os
import requests
from tqdm import tqdm
import tarfile
from pathlib import Path

import torch


LEGACY_MODEL_FILES_DIR = Path("./model_files")
LEGACY_DATA_DIR = Path("./data")
DEFAULT_MODEL_CONFIG = {
    "nlayers": 4,
    "output_dim": 1280,
    "d_hid": 5120,
    "token_dim": 5120,
}
MODEL_WEIGHT_CANDIDATES = (
    "model.pt",
    "pytorch_model.bin",
    "model.safetensors",
    "4layer_model.torch",
    "33layer_model.torch",
)
TOKEN_FILE_CANDIDATES = ("all_tokens.torch", "tokens.pt", "tokens.torch")
MISC_FILE_CANDIDATES = {
    "spec_chrom_csv_path": ("species_chrom.csv",),
    "offset_pkl_path": ("species_offsets.pkl",),
    "token_file": TOKEN_FILE_CANDIDATES,
    "new_species_csv_path": ("new_species_protein_embeddings.csv",),
}
MISC_DIR_CANDIDATES = {
    "protein_embeddings_dir": ("protein_embeddings",),
}


def normalize_path(path):
    if path is None:
        return None
    return str(Path(path).expanduser())


def _find_named_entry(base_dir, candidates, entry_type="file"):
    base_path = Path(base_dir).expanduser()
    if not base_path.exists():
        return None

    want_dir = entry_type == "dir"
    for candidate in candidates:
        direct_match = base_path / candidate
        if direct_match.exists() and direct_match.is_dir() == want_dir:
            return str(direct_match)

    for candidate in candidates:
        for match in sorted(base_path.rglob(candidate)):
            if match.is_dir() == want_dir:
                return str(match)

    return None


def _find_weight_file(model_dir):
    model_path = _find_named_entry(model_dir, MODEL_WEIGHT_CANDIDATES, entry_type="file")
    if model_path is not None:
        return model_path

    base_path = Path(model_dir).expanduser()
    for pattern in ("*.pt", "*.bin", "*.torch", "*.safetensors"):
        for match in sorted(base_path.rglob(pattern)):
            if match.name.endswith(".index.json"):
                continue
            return str(match)
    return None


def load_model_config(model_dir):
    config_path = _find_named_entry(model_dir, ("config.json",), entry_type="file")
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        key: config[key]
        for key in DEFAULT_MODEL_CONFIG
        if key in config
    }


def resolve_uce_runtime_args(args):
    args.hf_model_dir = normalize_path(getattr(args, "hf_model_dir", None))
    args.hf_misc_dir = normalize_path(getattr(args, "hf_misc_dir", None))
    args.model_loc = normalize_path(getattr(args, "model_loc", None))
    args.adata_path = normalize_path(getattr(args, "adata_path", None))

    if args.hf_model_dir is not None and not Path(args.hf_model_dir).is_dir():
        raise FileNotFoundError(f"HF model dir does not exist: {args.hf_model_dir}")
    if args.hf_misc_dir is not None and not Path(args.hf_misc_dir).is_dir():
        raise FileNotFoundError(f"HF misc dir does not exist: {args.hf_misc_dir}")

    model_config = load_model_config(args.hf_model_dir) if args.hf_model_dir else {}
    for key, default_value in DEFAULT_MODEL_CONFIG.items():
        current_value = getattr(args, key, None)
        if current_value is None:
            setattr(args, key, model_config.get(key, default_value))

    if args.model_loc is None:
        if args.hf_model_dir is not None:
            args.model_loc = _find_weight_file(args.hf_model_dir)
            if args.model_loc is None:
                raise FileNotFoundError(
                    f"Could not find a model checkpoint under HF model dir: {args.hf_model_dir}"
                )
        else:
            default_model_name = "33layer_model.torch" if args.nlayers == 33 else "4layer_model.torch"
            args.model_loc = str(LEGACY_MODEL_FILES_DIR / default_model_name)

    for attr_name, candidates in MISC_FILE_CANDIDATES.items():
        current_value = normalize_path(getattr(args, attr_name, None))
        if current_value is not None:
            setattr(args, attr_name, current_value)
            continue

        if args.hf_misc_dir is not None:
            resolved_path = _find_named_entry(args.hf_misc_dir, candidates, entry_type="file")
            if resolved_path is None and attr_name != "new_species_csv_path":
                raise FileNotFoundError(
                    f"Could not find {candidates[0]} under HF misc dir: {args.hf_misc_dir}"
                )
            setattr(args, attr_name, resolved_path)
        else:
            legacy_default = LEGACY_MODEL_FILES_DIR / candidates[0]
            setattr(args, attr_name, str(legacy_default))

    for attr_name, candidates in MISC_DIR_CANDIDATES.items():
        current_value = normalize_path(getattr(args, attr_name, None))
        if current_value is not None:
            setattr(args, attr_name, current_value)
            continue

        if args.hf_misc_dir is not None:
            resolved_dir = _find_named_entry(args.hf_misc_dir, candidates, entry_type="dir")
            if resolved_dir is None:
                raise FileNotFoundError(
                    f"Could not find {candidates[0]}/ under HF misc dir: {args.hf_misc_dir}"
                )
            setattr(args, attr_name, resolved_dir)
        else:
            setattr(args, attr_name, str(LEGACY_MODEL_FILES_DIR / candidates[0]))

    if args.adata_path is None and args.hf_misc_dir is not None:
        args.adata_path = _find_named_entry(args.hf_misc_dir, ("10k_pbmcs_proc.h5ad",), entry_type="file")

    return args


def load_model_state_dict(model_path):
    checkpoint_path = Path(model_path).expanduser()
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Checkpoint is stored as .safetensors. Install `safetensors` or pass a .pt/.bin/.torch checkpoint."
            ) from exc
        return load_file(str(checkpoint_path))

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested_state_dict = checkpoint.get(key)
            if isinstance(nested_state_dict, dict):
                return nested_state_dict
        return checkpoint

    raise TypeError(f"Unsupported checkpoint format in {model_path}")


def get_shapes_dict(dataset_path):
    shapes_dict = {}
    datasets_df = pd.read_csv(dataset_path)
    sorted_dataset_names = sorted(datasets_df["names"])

    for name in sorted_dataset_names:
        shapes_dict[name] = (int(datasets_df.set_index("names").loc[name]["num_cells"]), 8000)

    shapes_dict["dev_immune_mouse"] = (443697, 4786)
    shapes_dict["dev_immune_human"] = (34009, 5566)
    shapes_dict["intestinal_tract_human"] =  (69668, 5192)
    shapes_dict["gtex_human"] =  (18511, 7109)
    shapes_dict["gut_endoderm_mouse"] =  (113043, 6806)
    shapes_dict["luca"] =  (249591, 7196)
    shapes_dict.update({
     "madissoon_novel_lung":(190728, 8000),
     'flores_cerebellum_human': (20232, 8000),
     'osuch_gut_human': (272310, 8000),
     'msk_ovarian_human': (929690, 8000),
     'htan_vmuc_dis_epi_human': (65084, 8000),
     'htan_vmuc_val_epi_human': (57564, 8000),
     'htan_vmuc_non_epi_human': (9099, 8000),
     'hao_pbmc_3p_human': (161764, 8000),
     'hao_pbmc_5p_human': (49147, 8000),
     'gao_tumors_human': (36111, 8000),
     'swabrick_breast_human': (92427, 8000),
     'wu_cryo_tumors_human': (105662, 8000),
     'cell_line_het_human': (53513, 8000),
     'bi_allen_metastasis_human': (27787, 8000),
     'zheng68k_human': (68579, 8000),
     'zheng68k_12k_human': (68579, 12000),
     'mouse_embryo_ct': (153597, 12000),
     "regev_gtex_heart": (36574, 8000),
     "tabula_sapiens_heart": (11505, 8000),
     "10k_pbmcs":(11990, 12000),
     "epo_ido":(35834,12000),
     'tabula_sapiens_kidney': (9641, 8000),
     'tabula_microcebus_kidney': (14592, 8000),
     'tabula_muris_kidney': (2781, 8000),
     'tabula_muris_senis_kidney': (19610, 8000),
      'immune_human': (33506, 8000)
                       })

    shapes_dict["zyl_sanes_glaucoma_pig"] = (5901, 6819)
    shapes_dict["parkinsons_macaF"] = (1062, 5103)

    for row in datasets_df.iterrows():
        ngenes = row[1].num_genes
        ncells = row[1].num_cells
        name = row[1].names
        if not np.isnan(ngenes):
            shapes_dict[name] = (int(ncells), int(ngenes))

    return shapes_dict


def figshare_download(url, save_path):
    """
    Figshare download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """

    if os.path.exists(save_path):
        return
    else:
        # Check if directory exists
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        print("Downloading " + save_path + " from " + url + " ..." + "\n")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB',
                            unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    # If the downloaded filename ends in tar.gz then extraact it
    if save_path.endswith(".tar.gz"):
       with tarfile.open(save_path) as tar:
            tar.extractall(path=os.path.dirname(save_path))
            print("Done!")
