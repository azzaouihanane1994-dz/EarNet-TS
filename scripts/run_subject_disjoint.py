"""Run custom subject-disjoint evaluation with unseen-identity gallery validation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def validate_protocol(protocol, check_files=False):
    if not protocol.get('subject_disjoint') or not protocol.get('independent_test'):
        raise ValueError('A subject-disjoint independent protocol is required')
    identities, images = {}, {}
    for split in ('train', 'val', 'test'):
        rows = protocol['items'][split]
        identities[split] = {identity for _, identity in rows}
        images[split] = {image for image, _ in rows}
        for image in images[split]:
            path = Path(image)
            if path.is_absolute() or not path.parts or path.parts[0] != 'data' or '..' in path.parts:
                raise ValueError(f'Invalid dataset path: {image}')
            if check_files and not (ROOT / path).is_file():
                raise FileNotFoundError(image)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if identities[a] & identities[b] or images[a] & images[b]:
            raise ValueError(f'Overlapping identities or images: {a}/{b}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['ami', 'awe', 'earvn10', 'uerc2019'],
                        default=['ami', 'awe', 'earvn10', 'uerc2019'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--variants', nargs='+', choices=['earnet', 'rgb', 'single_fusion_matched'],
                        default=['earnet'])
    parser.add_argument('--output', type=Path, default=Path('runs/subject_disjoint'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--execute', action='store_true', help='Train and evaluate; otherwise inspect only')
    parser.add_argument('--check', action='store_true', help='Also check that dataset files exist')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'runs') or output == ROOT / 'runs':
        parser.error('Output must be a subdirectory of runs/')
    config = json.loads((ROOT / 'configs/subject_disjoint.json').read_text(encoding='utf-8'))
    jobs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            protocol_path = ROOT / 'protocols/subject_disjoint' / f'{dataset}_seed{seed}.json'
            if not protocol_path.is_file():
                parser.error(f'No recorded subject-disjoint membership for {dataset}, seed {seed}')
            protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
            validate_protocol(protocol, args.check or args.execute)
            for variant in args.variants:
                overrides = {'architecture': 'rgb', 'edh': False, 'onsa': False} if variant == 'rgb' else (
                    {'architecture': 'single_fusion', 'edh': False, 'onsa': False}
                    if variant == 'single_fusion_matched' else {})
                jobs.append({'id': f'{dataset}_{variant}_seed{seed}', 'collection': dataset,
                             'seed': seed, 'variant': variant, 'group': 'core',
                             'config': {**config, **overrides, 'device': args.device}})
    if not args.execute:
        print(json.dumps({'protocol': config['protocol'], 'output': output.relative_to(ROOT).as_posix(),
                          'jobs': jobs}, indent=2))
        return
    os.chdir(ROOT)
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['TORCH_HOME'] = str(ROOT / 'weights/imagenet')
    os.environ['YOLO_CONFIG_DIR'] = str(output / 'detector_settings')
    import torch
    from earnet_ts.experiments.training import BASE, train_job
    output.mkdir(parents=True, exist_ok=True)
    (output / 'protocols').mkdir(exist_ok=True)
    for job in jobs:
        destination = output / 'protocols' / f"{job['collection']}_seed{job['seed']}.json"
        source = ROOT / 'protocols/subject_disjoint' / destination.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError('Resume split membership differs from the recorded protocol')
        shutil.copyfile(source, destination)
        job['config'] = {**BASE, **job['config']}
        if args.device == 'auto':
            job['config']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        if (output / 'runs' / job['id'] / 'metrics.json').exists():
            raise ValueError(f"Completed output already exists: {job['id']}; select another output directory")
        train_job(ROOT, output, job)


if __name__ == '__main__':
    main()
