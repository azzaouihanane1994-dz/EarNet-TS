# Subject-disjoint results

`metrics.json` records the five-split aggregate results presented in the article: Rank-1 mean, sample standard deviation and EER. Measurements were supplied by the authors from a separate workstation. These aggregates are distinct from individual local training runs.

The custom 70/10/20 protocol separates training, validation and test identities. One image per evaluation identity forms the gallery; remaining images are probes. AWE subject-disjoint evaluation is separate from its official fixed 600/400 identity-sharing benchmark. UERC-2019 uses the evaluated 330-identity subset.

Recorded local split memberships for seeds 42--46 are in `protocols/subject_disjoint/`. AWE manifests reference the cached AWE-member images under `data/uerc2019/PartA/`; these prepared images differ from the images under `data/awe/` used by the fixed benchmark launcher. The two input sets must not be substituted silently.

The subject-disjoint launcher replays these manifests and writes new outputs under `runs/subject_disjoint/`. The aggregate file does not assert that these local manifests are the per-split records from the separate workstation.
