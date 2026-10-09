# Datasets

EarNet-TS evaluates ear recognition on AMI, AWE, EarVN1.0 and UERC-2019. Dataset paths in the run configurations are relative to the repository root.

## Acquisition and directory structure

Obtain the datasets from their respective providers and arrange the inputs as follows:

| Dataset | Dataset provider | Directory structure |
|---|---|---|
| AMI | [AMI Ear Database](https://webctim.ulpgc.es/research_works/ami_ear_database/) | `data/ami/<image>.jpg` |
| AWE | [AWE datasets](http://awe.fri.uni-lj.si/datasets.html) | `data/awe/<subject>/<image>.png` |
| EarVN1.0 | [EarVN1.0 dataset](https://data.mendeley.com/datasets/yws3v3mwx3/4) | `data/earvn10/<subject>/<image>.jpg.png` |
| UERC-2019 | [UERC dataset information](http://awe.fri.uni-lj.si/datasets.html) | `data/uerc2019/PartA/<subject>/<image>.png.png` and `data/uerc2019/PartB/<subject>/<image>.png.png` |

The evaluated AMI inputs are photographs. The evaluated AWE, EarVN1.0 and UERC-2019 inputs are cached ear crops. The double extensions above are the actual filenames referenced by the saved partitions. Preserve those names when using the recorded splits.

`manifest.json` records repository-relative filenames, file sizes and SHA-256 hashes for the evaluated inputs. These hashes refer to the prepared images; downloaded photographs must undergo the corresponding crop preparation before their hashes can match. Renaming an original photograph does not reproduce a crop.

## AWE benchmark split

AWE is treated as the separate official AWE benchmark. Its `train.txt` and `test.txt` lists define 600 training images and 400 test images. The AWE launcher preserves this membership for every seed.

Each index `i` maps to:

- Subject: `(i - 1) // 10 + 1`, formatted with three digits.
- Image: `(i - 1) % 10 + 1`, formatted with two digits.

For example, index `1` corresponds to `data/awe/001/01.png`. All 600 training images are used for training. Training-set monitoring is separate from final evaluation on the 400 test images; the checkpoint is selected at the predetermined final epoch.

## Recorded partitions and validation

AMI, EarVN1.0 and UERC-2019 seed 43/44 runs use the image memberships recorded in `results/repeats/<run>/splits.json`. The `--new-split` option generates a seed-dependent partition for those datasets. AWE always uses its fixed benchmark lists.

From the repository root, check the input files and required weights with:

```bash
python scripts/check_release.py --local-data
```

## Dataset terms

Follow each provider's access, usage and redistribution terms. Dataset licenses are separate from the MIT license for the authors' code.
