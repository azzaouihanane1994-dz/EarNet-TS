"""Launch an original full-model v3 run with portable paths."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_configuration(dataset, seed, output):
    name = f'awe_fixed_seed{seed}' if dataset == 'awe' else f'{dataset}_seed{seed}'
    path = ROOT / 'configs' / f'{name}.json'
    template = path if path.exists() else ROOT / 'configs' / (
        'awe_fixed_seed43.json' if dataset == 'awe' else f'{dataset}_seed43.json')
    cfg = json.loads(template.read_text(encoding='utf-8'))
    cfg['seed'] = seed
    if seed != 44 and dataset == 'earvn10':
        cfg['epochs'], cfg['fp16'] = 100, True
    cfg['work_dir'] = output.relative_to(ROOT).as_posix()
    cfg['crop_cache_dir'] = (output / 'crop_cache').relative_to(ROOT).as_posix()
    for key in ('dataset_dir', 'pretrained_ckpt', 'ear_detector_weights'):
        if cfg.get(key):
            cfg[key] = (ROOT / cfg[key]).resolve().relative_to(ROOT).as_posix()
    return cfg, name


def install_saved_split(module, path):
    rows = json.loads(path.read_text(encoding='utf-8'))
    splits = {k: [(Path(image), identity) for image, identity in items]
              for k, items in rows.items()}
    for items in splits.values():
        for image, _ in items:
            if not (ROOT / image).is_file():
                raise FileNotFoundError(image)
    labels = {pid: i for i, pid in enumerate(sorted(
        {pid for items in splits.values() for _, pid in items}))}
    module.split_dataset = lambda cfg: (
        splits['train'], splits['val'], splits['test'], labels)


def prepare_module(dataset, cfg, seed, output, replay):
    source_dataset = 'uerc2019' if dataset == 'awe' else dataset
    path = ROOT / 'models' / source_dataset / 'novel_model_v3.py'
    spec = importlib.util.spec_from_file_location(f'full_v3_{dataset}', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CONFIG = copy.deepcopy(cfg)
    import torch
    module.CONFIG['device'] = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device'] == 'auto' else cfg['device']
    if dataset != 'ami':
        # These inputs are already cropped; retain their full image geometry.
        module.EarVNDataset._crop_and_cache = lambda self, image: module.safe_load_image(image)
        module.CONFIG['use_yolo_crop'] = False
        module.CONFIG['force_recache'] = False
    if dataset == 'awe':
        from awe_fixed_protocol import apply_fixed_awe
        apply_fixed_awe(module, output, ROOT)
    elif replay:
        install_saved_split(module, replay)
    if dataset == 'earvn10' and seed == 44:
        from original_v3_numerics import repair_angular_boundary
        repair_angular_boundary(module)
        original_step = torch.optim.AdamW.step
        def clipped_step(optimizer, *args, **kwargs):
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']], 5.0)
            return original_step(optimizer, *args, **kwargs)
        torch.optim.AdamW.step = clipped_step
    # Retain each source's batching and scoring behavior.
    if hasattr(module, 'save_full_verification_artifacts'):
        export = module.save_full_verification_artifacts
        def export_summary(*args, **kwargs):
            result = export(*args, **kwargs)
            if result is not None:
                return result
            candidates = list(output.rglob('verification_summary_full.json'))
            if len(candidates) != 1:
                raise RuntimeError('Verification export did not provide a unique summary')
            return json.loads(candidates[0].read_text(encoding='utf-8'))
        module.save_full_verification_artifacts = export_summary
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['ami', 'awe', 'earvn10', 'uerc2019'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--new-split', action='store_true', help='Generate the original seed-dependent image split instead of replaying saved membership')
    p.add_argument('--import-check', action='store_true', help='Import and configure the model without training')
    args = p.parse_args()
    output = (ROOT / (args.output or Path('runs') / f'{args.dataset}_seed{args.seed}')).resolve()
    if ROOT / 'runs' not in output.parents:
        p.error('Outputs must be inside runs/')
    if output.exists() and any(output.iterdir()):
        p.error('Output directory must be empty')
    cfg, name = load_configuration(args.dataset, args.seed, output)
    replay = ROOT / 'results/repeats' / name / 'splits.json'
    replay = replay if replay.exists() and not args.new_split else None
    if args.dry_run:
        print(json.dumps({'dataset': args.dataset, 'configuration': cfg,
                          'saved_split': replay.relative_to(ROOT).as_posix() if replay else None}, indent=2))
        return
    os.chdir(ROOT)
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['YOLO_CONFIG_DIR'] = str(output / 'detector_settings')
    os.environ['TORCH_HOME'] = str(ROOT / 'weights' / 'imagenet')
    for key in ('dataset_dir', 'pretrained_ckpt', 'ear_detector_weights'):
        if cfg.get(key) and not Path(cfg[key]).exists():
            p.error(f'Missing {key}: {cfg[key]}')
    output.mkdir(parents=True, exist_ok=True)
    module = prepare_module(args.dataset, cfg, args.seed, output, replay)
    (output / 'configuration.json').write_text(json.dumps(module.CONFIG, indent=2), encoding='utf-8')
    if args.import_check:
        print('Model import and split configuration passed; no training executed.')
        return
    module.main()


if __name__ == '__main__':
    main()
