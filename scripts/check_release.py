"""Check original sources, partitions and published full-model metrics."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {
    'ami': (97.50, .998567, 2.869, 97.00, 97.47),
    'awe': (85.50, .939225, 12.000, 68.00, 84.00),
    'earvn10': (93.50, .986854, 4.168, 89.51, 93.54),
    'uerc2019': (78.33, .944360, 12.424, 48.18, 77.12),
}
REPEATS = {
    ('ami', 43): (98.50, 93.00, .9941, 3.00),
    ('ami', 44): (99.00, 93.00, .9981, 2.00),
    ('awe', 43): (85.00, 56.67, .9468, 11.50),
    ('awe', 44): (87.25, 61.67, .9467, 12.17),
    ('earvn10', 43): (92.01, 83.02, .9805, 5.69),
    ('earvn10', 44): (92.46, 85.61, .9796, 5.52),
    ('uerc2019', 43): (73.64, 46.97, .9404, 13.03),
    ('uerc2019', 44): (72.88, 48.18, .9378, 13.94),
}

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-data', action='store_true', help='Require all local split images and weight files')
    p.add_argument('--hashes', action='store_true', help='Verify all packaged public artifact hashes')
    p.add_argument('--scores', action='store_true', help='Recalculate verification AUC/EER from stored score arrays')
    args = p.parse_args()
    for row in read('models/source_manifest.json'):
        assert digest(ROOT / row['path']) == row['sha256'], row['path']
    for row in read('results/reference/metrics.json'):
        expected = REFERENCE[row['dataset']]
        actual = (row['classifier_accuracy_percent'], row['verification']['auc'],
                  row['verification']['eer_percent'], row['identification']['ranks_percent']['1'],
                  row['centroid_matching']['topk_percent']['1'])
        assert tuple(round(x, n) for x, n in zip(actual, [2, 6, 3, 2, 2])) == expected, row['dataset']
    repeated = read('results/repeats/metrics.json')
    assert len(repeated) == 8
    for row in repeated:
        actual = (row['classifier_accuracy_percent'], row['gallery_rank1_percent'],
                  row['distinct_pair_auc'], row['distinct_pair_eer_percent'])
        assert tuple(round(x, n) for x, n in zip(actual, [2, 2, 4, 2])) == REPEATS[row['dataset'], row['seed']]
    for path in (ROOT / 'results').rglob('splits.json'):
        rows = json.loads(path.read_text(encoding='utf-8'))
        assert set(rows) == {'train', 'val', 'test'}
        if args.local_data:
            for items in rows.values():
                for image, _ in items:
                    assert (ROOT / image).is_file(), image
    train = list(map(int, (ROOT / 'data/awe/train.txt').read_text().split()))
    test = list(map(int, (ROOT / 'data/awe/test.txt').read_text().split()))
    assert len(set(train)) == len(train) == 600
    assert len(set(test)) == len(test) == 400
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(1, 1001))
    for path in (ROOT / 'protocols/subject_disjoint').glob('*_seed*.json'):
        protocol = json.loads(path.read_text(encoding='utf-8'))
        assert protocol['subject_disjoint'] and protocol['independent_test'], path
        ids = {s: {pid for _, pid in items} for s, items in protocol['items'].items()}
        images = {s: {image for image, _ in items} for s, items in protocol['items'].items()}
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            assert not ids[a] & ids[b] and not images[a] & images[b], path
        for items in protocol['items'].values():
            for image, _ in items:
                path = Path(image)
                assert not path.is_absolute() and path.parts[0] == 'data' and '..' not in path.parts, image
                if args.local_data:
                    assert (ROOT / image).is_file(), image
    aggregates = read('results/subject_disjoint/metrics.json')['results']
    assert len(aggregates) == 12
    proposed = {row['dataset']: (row['mean_rank1_percent'], row['sample_sd_rank1_percentage_points'], row['eer_percent'])
                for row in aggregates if row['configuration'] == 'Proposed configuration'}
    assert proposed == {'AMI': (90.83, 1.82, 5.71), 'AWE': (41.11, 2.75, 21.56),
                        'EarVN1.0': (49.49, 1.43, 16.75), 'UERC-2019': (30.85, 2.36, 22.49)}
    assert all(row['splits'] == 5 for row in aggregates)
    if args.hashes:
        for row in read('provenance/artifacts.json'):
            assert digest(ROOT / row['path']) == row['sha256'], row['path']
    if args.local_data:
        for row in read('weights/manifest.json'):
            assert (ROOT / row['path']).is_file(), row['path']
    if args.scores:
        import numpy as np
        from sklearn.metrics import roc_auc_score, roc_curve
        for ds in REFERENCE:
            stored = np.load(ROOT / 'results/reference' / ds / 'scores.npz')
            genuine, impostor = stored['genuine_scores'], stored['impostor_scores']
            y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
            scores = np.r_[genuine, impostor]
            auc = roc_auc_score(y, scores)
            far, tar, _ = roc_curve(y, scores)
            gap = far - (1 - tar)
            upper = int(np.flatnonzero(gap >= 0)[0])
            if upper == 0 or gap[upper] == 0:
                eer = far[upper]
            else:
                lower = upper - 1
                weight = -gap[lower] / (gap[upper] - gap[lower])
                eer = far[lower] + weight * (far[upper] - far[lower])
            assert round(auc, 6) == REFERENCE[ds][1], ds
            assert round(100 * eer, 3) == REFERENCE[ds][2], ds
    print('Release checks passed: original sources, reference/repeated results, official AWE lists, subject-disjoint splits and article aggregates.')

if __name__ == '__main__':
    main()
