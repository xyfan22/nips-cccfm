# Third-party provenance and licenses

The root MIT license applies to our code, documentation, and numerical
analysis outputs. It does not
relicense sources under `baselines/`, upstream data, or model checkpoints.
Original notices and upstream authorship are retained.

| Bundled project | Source | License/notice in the bundled snapshot |
|---|---|---|
| CellFlow | https://github.com/theislab/cellflow | Bundled `LICENSE`: MIT; `pyproject.toml`: PolyForm-Noncommercial-1.0.0. Both upstream declarations are retained; check upstream terms before reuse. |
| GEARS | https://github.com/snap-stanford/GEARS | MIT |
| CPA | https://github.com/theislab/cpa | BSD-3-Clause |
| CellOT | https://github.com/bunnech/cellot | BSD-3-Clause |
| scGPT | https://github.com/bowang-lab/scGPT | MIT |
| scFoundation | https://github.com/biomap-research/scFoundation | Apache-2.0 |
| UCE | https://github.com/snap-stanford/UCE | Bundled code: MIT; checkpoint terms must be checked separately |
| Geneformer | https://huggingface.co/ctheodoris/Geneformer | Upstream model-card metadata declares Apache-2.0; its original attribution is retained |
| STATE | https://github.com/ArcInstitute/state | CC-BY-NC-SA-4.0 |
| STACK | https://github.com/ArcInstitute/stack | CC-BY-NC-SA-4.0 |
| PerturbGen | https://github.com/theislab/perturbgen | Bundled code: MIT |
| Systema | https://doi.org/10.5281/zenodo.15746994 | Official v1.0.0 software archive: CC-BY-4.0 |

## Systema archive attribution

The authors' [v1.0.0 software archive](https://zenodo.org/records/15746995)
declares CC-BY-4.0. All 14 bundled executable Python/shell files were compared
byte-for-byte with that archive; they match. The local README files have
documentation changes, and `notes.md` is a local reference note.
See the [CC-BY-4.0 terms](https://creativecommons.org/licenses/by/4.0/legalcode).
Credit: Ramon Vinas Torne and the Systema contributors, *Systema: a framework
for evaluating genetic perturbation response prediction beyond systematic
variation*, Nature Biotechnology, DOI 10.1038/s41587-025-02777-8.

## Data and weights

No upstream model checkpoint or single-cell expression matrix is
redistributed. Public biological sources are identified in
`data/sources.json`. Their provider terms apply independently of this
repository's code license. Weight licensing can differ from code licensing;
the historical links in `download_list.md` are discovery notes, not a
replacement for the checkpoint's current model card and license.
