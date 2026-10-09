"""Run full-model repetitions sequentially, one process per seed."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=['ami', 'awe', 'earvn10', 'uerc2019'], default=['ami', 'awe', 'earvn10', 'uerc2019'])
    p.add_argument('--seeds', nargs='+', type=int, default=[43, 44])
    p.add_argument('--execute', action='store_true', help='Start training; otherwise inspect each run configuration')
    p.add_argument('--new-split', action='store_true')
    args = p.parse_args()
    runner = Path(__file__).with_name('run_v3.py')
    for dataset in args.datasets:
        for seed in args.seeds:
            command = [sys.executable, str(runner), '--dataset', dataset, '--seed', str(seed)]
            if not args.execute:
                command += ['--dry-run']
            if args.new_split:
                command += ['--new-split']
            subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
