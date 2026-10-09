# EarNet-TS

Full-model implementations and experimental evidence for **EarNet-TS: A Two-Stream Contour-Enhanced Attention Network with ECA and Scheduled ArcFace for Ear Biometrics**.

EarNet-TS combines RGB appearance and contour information for ear recognition. The model learns 512-dimensional embeddings and uses scheduled ArcFace during training. Evaluation covers AMI, AWE, EarVN1.0 and UERC-2019.

## Model architecture

- **RGB and contour streams:** ResNet-50 feature extractors process ear appearance and structural information.
- **Feature fusion and ECA:** the streams are fused with efficient channel attention to weight their combined features.
- **CBAM:** channel and spatial attention refine the deeper feature representation.
- **Embedding head:** the network produces normalized 512-dimensional descriptors for cosine-similarity matching.
- **EDH:** an auxiliary embedding-distribution head operates on detached fused features and is optimized separately from the backbone.
- **ONSA:** occlusion/noise-aware auxiliary supervision uses an occluder encoder and a detach-based adversarial objective.
- **Scheduled ArcFace:** the angular margin and logit scale follow the schedules defined in each configuration.

AMI, EarVN1.0 and UERC-2019 use their dataset-specific v3 implementations. AWE uses the UERC-2019 v3 architecture with the official AWE membership lists. The launchers configure data locations, seeds and execution settings through repository-relative paths.

## Setup

Use a dedicated Python environment and install `requirements.txt`. Model imports require PyTorch, torchvision and image-processing dependencies. The dependency versions and environment information are recorded in `requirements.txt` and `docs/import_environment.json`. Select a PyTorch/CUDA installation compatible with your machine.

```bash
python -m pip install -r requirements.txt
python scripts/check_release.py
python scripts/run_v3.py --dataset awe --seed 43 --dry-run
python scripts/run_multiple.py --datasets ami awe earvn10 uerc2019 --seeds 43 44
```

The multi-run command above only inspects configurations. Start sequential training explicitly:

```bash
python scripts/run_multiple.py --datasets ami awe earvn10 uerc2019 --seeds 43 44 --execute
```

Run outputs are saved under `runs/` in a new, empty directory. For seeds 43/44, the launcher replays the recorded image memberships. `--new-split` invokes the original seed-dependent split generator for AMI, EarVN1.0 and UERC-2019. AWE always preserves its supplied 600/400 benchmark lists. Use `--seeds` to select additional random seeds.

## Full-model reference results

| Dataset | Classifier accuracy (%) | AUC | EER (%) | Gallery Rank-1 (%) | Training-centroid Top-1 (%) |
|---|---:|---:|---:|---:|---:|
| AMI | 97.50 | 0.998567 | 2.869 | 97.00 | 97.47 |
| AWE | 85.50 | 0.939225 | 12.000 | 68.00 | 84.00 |
| EarVN1.0 | 93.50 | 0.986854 | 4.168 | 89.51 | 93.54 |
| UERC-2019 | 78.33 | 0.944360 | 12.424 | 48.18 | 77.12 |

`results/reference/metrics.json` and the per-dataset metric files preserve higher precision. Classifier accuracy, single-test-image gallery matching and training-centroid retrieval are different tasks. Verification uses distinct genuine/impostor pairs and interpolation of the empirical FAR/FRR crossing.

## Repeated runs

| Dataset | Seed | Classifier (%) | Gallery Rank-1 (%) | AUC | EER (%) |
|---|---:|---:|---:|---:|---:|
| AMI | 43 | 98.50 | 93.00 | 0.9941 | 3.00 |
| AMI | 44 | 99.00 | 93.00 | 0.9981 | 2.00 |
| AWE | 43 | 85.00 | 56.67 | 0.9468 | 11.50 |
| AWE | 44 | 87.25 | 61.67 | 0.9467 | 12.17 |
| EarVN1.0 | 43 | 92.01 | 83.02 | 0.9805 | 5.69 |
| EarVN1.0 | 44* | 92.46 | 85.61 | 0.9796 | 5.52 |
| UERC-2019 | 43 | 73.64 | 46.97 | 0.9404 | 13.03 |
| UERC-2019 | 44 | 72.88 | 48.18 | 0.9378 | 13.94 |

EarVN1.0 seed 44* uses FP32, an angular square-root argument floor of 1e-7, gradient-norm clipping at 5, and 95 completed epochs with the validation-selected epoch-92 checkpoint. It is not a strictly seed-only repeat. Other recorded EarVN1.0 repetitions use their own 100-epoch budget, distinct from the 150-epoch reference configuration. AWE repetitions use 70 epochs and a final-epoch checkpoint; the training loader also supplies the monitoring loader and is not an independent validation set. Their classifier scores include the target angular margin. These settings are not pooled as identical conditions.

## Evaluation protocols

- Reference AMI has validation/test image overlap. Its scores do not establish independent-test generalization.
- Reference AWE uses a 700/100/200 identity-sharing partition; the repeated AWE runs use the official fixed 600/400 partition. These are distinct protocols.
- Evaluated UERC-2019 comprises 3,300 images from 330 identities in Parts A and B, rather than the official challenge evaluation. AWE shares images with UERC; the two collections are not independent.
- Detector training identity membership and transfer-checkpoint training membership are undocumented. Exclusion of recognition test subjects from those preprocessing/initialization stages cannot be established.
- AMI training-centroid matching excludes two queries without matching centroids, leaving 198 known queries.
- The detached EDH does not directly backpropagate its loss to the backbone. Triplet coefficient is zero in the evaluated reference objective. The EarVN1.0 reference retains its original cached angular coefficients.

## Data and weights

Dataset acquisition links and preparation details are provided in `data/README.md`. Official AWE train/test membership lists define the fixed 600/400 protocol. Arrange the image directories as described there and use the relative paths in `configs/`.

Weight locations and SHA-256 hashes are documented in `weights/README.md` and `weights/manifest.json`. Place the required detector, ImageNet and transfer weights at those locations before launching a run.

## License and publication

The authors' code is licensed under the MIT license. Third-party datasets and weights retain their own terms. Repository publication instructions are provided in `docs/PUBLISHING.md`.
