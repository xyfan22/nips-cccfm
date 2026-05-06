"""Helper functions for loading pretrained gene embeddings."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from scanpy import AnnData
import numpy as np
import pandas as pd


UCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_DIR = UCE_ROOT / "model_files" / "protein_embeddings"
DEFAULT_EXTRA_SPECIES_CSV = UCE_ROOT / "model_files" / "new_species_protein_embeddings.csv"


def load_extra_species_paths(extra_species_csv_path: Optional[str] = None) -> Dict[str, Path]:
    csv_path = Path(extra_species_csv_path).expanduser() if extra_species_csv_path else DEFAULT_EXTRA_SPECIES_CSV
    if not csv_path.exists():
        return {}

    extra_species = pd.read_csv(csv_path).set_index("species").to_dict()["path"]
    resolved_paths = {}
    for species, path in extra_species.items():
        species_path = Path(path).expanduser()
        if not species_path.is_absolute():
            species_path = (csv_path.parent / species_path).resolve()
        resolved_paths[species] = species_path
    return resolved_paths


def get_species_to_embedding_paths(
    embedding_dir: Optional[str] = None,
    extra_species_csv_path: Optional[str] = None,
) -> Dict[str, Path]:
    base_embedding_dir = Path(embedding_dir).expanduser() if embedding_dir else DEFAULT_EMBEDDING_DIR
    species_to_path = {
        'human': base_embedding_dir / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        'mouse': base_embedding_dir / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
        'frog': base_embedding_dir / 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt',
        'zebrafish': base_embedding_dir / 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt',
        "mouse_lemur": base_embedding_dir / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
        "pig": base_embedding_dir / 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt',
        "macaca_fascicularis": base_embedding_dir / 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt',
        "macaca_mulatta": base_embedding_dir / 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt',
    }
    species_to_path.update(load_extra_species_paths(extra_species_csv_path))
    return species_to_path


def load_gene_embeddings_adata(
    adata: AnnData,
    species: list,
    embedding_model: str,
    embedding_dir: Optional[str] = None,
    extra_species_csv_path: Optional[str] = None,
) -> Tuple[AnnData, Dict[str, torch.FloatTensor]]:
    """Loads gene embeddings for all the species/genes in the provided data.

    :param data: An AnnData object containing gene expression data for cells.
    :param species: Species corresponding to this adata
    
    :param embedding_model: The gene embedding model whose embeddings will be loaded.
    :return: A tuple containing:
               - A subset of the data only containing the gene expression for genes with embeddings in all species.
               - A dictionary mapping species name to the corresponding gene embedding matrix (num_genes, embedding_dim).
    """
    # Get species names
    species_names = species
    species_names_set = set(species_names)

    # Get embedding paths for the model
    if embedding_model != "ESM2":
        raise ValueError(f"Unsupported embedding model: {embedding_model}")
    species_to_gene_embedding_path = get_species_to_embedding_paths(
        embedding_dir=embedding_dir,
        extra_species_csv_path=extra_species_csv_path,
    )
    available_species = set(species_to_gene_embedding_path)

    # Ensure embeddings are available for all species
    if not (species_names_set <= available_species):
        raise ValueError(f'The following species do not have gene embeddings: {species_names_set - available_species}')

    # Load gene embeddings for desired species (and convert gene symbols to lower case)
    species_to_gene_symbol_to_embedding = {
        species: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(species_to_gene_embedding_path[species]).items()
        }
        for species in species_names
    }

    # Determine which genes to include based on gene expression and embedding availability
    genes_with_embeddings = set.intersection(*[
        set(gene_symbol_to_embedding)
        for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding.values()
    ])
    genes_to_use = {gene for gene in adata.var_names if gene.lower() in genes_with_embeddings}

    # Subset data to only use genes with embeddings
    adata = adata[:, adata.var_names.isin(genes_to_use)]

    # Set up dictionary mapping species to gene embedding matrix (num_genes, embedding_dim)
    species_to_gene_embeddings = {
        species_name: torch.stack([
            species_to_gene_symbol_to_embedding[species_name][gene_symbol.lower()]
            for gene_symbol in adata.var_names
        ])
        for species_name in species_names
    }

    return adata, species_to_gene_embeddings
