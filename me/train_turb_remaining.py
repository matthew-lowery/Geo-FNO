"""Recover unfinished turbulence experiments without repeating completed seeds."""

import argparse
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BASE = Path('/projects/bgcs/mlowery/operator-benchmarks')
IGNORED = {'data_root', 'model_folder', 'div_folder', 'project_name', 'gpu',
           'wandb', 'save', 'calc_div', 'require_ood', 'no_ood', 'resume',
           'eval_only', 'checkpoint_every'}
VARIABLE = {'seed', 'ntrain', 'div_loss_weight', 'div_loss'}
JOB_PATTERN = re.compile(r'(geo|trans)_forced_turb_s([123])_n(\d+)_l([\d.]+)$')


def options(command):
    result = {}
    for token in command:
        if token.startswith('--'):
            key, _, value = token[2:].partition('=')
            result[key.replace('-', '_')] = value if '=' in token else True
    result.setdefault('div_loss', False)
    return result


def key(model, config):
    coefficient = float(config['div_loss_weight']) if config.get('div_loss') else 0.
    return model, int(config['seed']), int(config['ntrain']), coefficient


def matches(expected, actual):
    def equal(a, b):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return a == b
    return all(k in actual and equal(v, actual[k]) for k, v in expected.items()
               if k not in IGNORED)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def complete(metrics, coefficient):
    required = ['test_loss', 'test_div/max_abs_interior', 'test_div/median_abs_interior']
    if coefficient == 0:
        required.append('ood_loss')
    try:
        return all(math.isfinite(float(metrics[k])) for k in required)
    except (KeyError, TypeError, ValueError):
        return False


def walltime(model, size):
    # Upper measured full-size epoch rates, rounded up (Sept 16 A100 logs).
    seconds = {'geo': 236., 'trans': 47.}[model] * size / 10000
    return math.ceil(1.35 * 500 * seconds / 3600 + 1)


def cached_runs(roots):
    for root in roots:
        for path in root.glob('**/files/wandb-metadata.json'):
            metadata = read_json(path)
            config = options(metadata.get('args', []))
            if config.get('dataset') != 'forced_turb':
                continue
            model = 'trans' if 'transolver' in metadata.get('program', '') else 'geo'
            yield model, config, read_json(path.with_name('wandb-summary.json'))


def queue_keys(required=False):
    if not shutil.which('squeue'):
        if required:
            raise RuntimeError('squeue unavailable; refusing submission without duplicate checks')
        print('Queue unchecked here; --submit requires a successful live queue check.', file=sys.stderr)
        return set()
    result = subprocess.run(['squeue', '-u', getpass.getuser(), '-h', '-o', '%200j'],
                            check=True, capture_output=True, text=True)
    keys = set()
    for name in result.stdout.splitlines():
        match = JOB_PATTERN.search(name.strip())
        if match:
            model, seed, size, coef = match.groups()
            keys.add((model, int(seed), int(size), float(coef)))
    return keys


def known_completed(jobs, roots):
    expected = {key(job['model'], job['config']): job['config'] for job in jobs}
    snapshot = read_json(HERE / 'turb_recovery_snapshot.json')
    done = set()
    for record in snapshot['completed']:
        model, seed, size, coef, _run = record
        identity = model, seed, size, coef
        config = {**snapshot['profiles'][model], 'seed': seed, 'ntrain': size,
                  'div_loss_weight': coef, 'div_loss': coef != 0}
        if identity in expected and matches(expected[identity], config):
            done.add(identity)
    for model, config, metrics in cached_runs(roots):
        identity = key(model, config)
        if (identity in expected and matches(expected[identity], config)
                and complete(metrics, identity[-1])):
            done.add(identity)
    return done


def plan(args):
    environment = dict(os.environ)
    environment['RAM_RESULTS_ROOT'] = str(args.results_root)
    command = ['bash', str(HERE / 'train_div.sh'), '--dry-run', '--dataset', 'forced_turb',
               '--model', args.model, '--phase', args.phase]
    for name in ('seed', 'ntrain'):
        value = getattr(args, name)
        if value is not None:
            command += ['--' + name, str(value)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    print(result.stderr, end='', file=sys.stderr)
    jobs = []
    for line in result.stdout.splitlines():
        phase, model, label, _hours, *command = shlex.split(line)
        config = options(command)
        jobs.append(dict(phase=phase, model=model, label='turbfix_' + label,
                         command=command, config=config, hours=walltime(model, int(config['ntrain']))))
    return jobs


def checkpoint_paths(job, roots):
    config = job['config']
    relative = Path(job['model']) / job['phase'] / ('lambda-' + config['div_loss_weight'])
    filename = f"forced_turb_{config['seed']}_{config['ntrain']}_7000.torch"
    return [root / relative / 'models' / filename for root in roots]


def select(jobs, done, active, roots):
    selected = []
    for job in jobs:
        identity = key(job['model'], job['config'])
        reason = 'completed' if identity in done else 'queued/running' if identity in active else None
        paths = checkpoint_paths(job, roots)
        for path in paths:
            record = read_json(path.with_suffix('.metrics.json'))
            if (matches(job['config'], record.get('config', {}))
                    and complete(record.get('metrics', {}), identity[-1])):
                reason = 'completed (local metrics)'
        if reason:
            print(f"Skip {job['label']}: {reason}", file=sys.stderr)
            continue
        checkpoint = next((p for p in paths if p.is_file()), None)
        if checkpoint and not any(t.startswith('--resume=') for t in job['command']):
            job['command'].append('--resume=' + str(checkpoint))
        selected.append(job)
    return selected


def plan_line(job):
    return f"{job['phase']} {job['model']} {job['label']} hours={job['hours']} " + shlex.join(job['command'])


def batch_script(job):
    return '\n'.join([
        '#!/bin/bash', '#SBATCH --partition=gpuA100x4', '#SBATCH --account=bgcs-delta-gpu',
        '#SBATCH --mem=32g', '#SBATCH --nodes=1', '#SBATCH --ntasks-per-node=1',
        '#SBATCH --cpus-per-task=1', '#SBATCH --gpus-per-node=1', '#SBATCH --constraint=scratch',
        f"#SBATCH --job-name={job['label']}", f"#SBATCH --time={job['hours']}:00:00",
        f'#SBATCH --output={HERE}/out/%x_%j.out', f'#SBATCH --error={HERE}/err/%x_%j.err',
        'set -euo pipefail', 'module purge', 'cd ' + shlex.quote(str(HERE)),
        'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        shlex.join(job['command']), '',
    ])


def run(args):
    jobs = plan(args)
    roots = args.wandb_root or [HERE / 'wandb', HERE / 'wandb_final']
    result_roots = [args.results_root, *args.previous_results_root]
    done = known_completed(jobs, roots)
    selected = select(jobs, done, queue_keys(args.submit), result_roots)
    print(f'{len(selected)} candidates; {len(jobs) - len(selected)} skipped.', file=sys.stderr)
    if args.submit or args.check:
        from check_training_plan import check_plan
        for job in selected:
            if not shutil.which(job['command'][0]):
                raise FileNotFoundError(f"Training Python missing: {job['command'][0]}")
        check_plan(plan_line(job) for job in selected)
    for job in selected:
        if not args.submit:
            print(plan_line(job))
            continue
        # Refresh before each submission: earlier jobs can finish during this launch.
        current_done = known_completed([job], roots)
        if not select([job], current_done, queue_keys(True), result_roots):
            continue
        for folder in ('out', 'err'):
            (HERE / folder).mkdir(exist_ok=True)
        subprocess.run(['sbatch'], input=batch_script(job), text=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--submit', action='store_true')
    parser.add_argument('--model', choices=['all', 'geo', 'trans'], default='all')
    parser.add_argument('--phase', choices=['all', 'div', 'baseline', 'forced'], default='all')
    parser.add_argument('--seed', type=int, choices=[1, 2, 3])
    parser.add_argument('--ntrain', type=int, choices=[100, 500, 1000, 5000, 7000, 10000])
    parser.add_argument('--wandb-root', type=Path, action='append')
    parser.add_argument('--results-root', type=Path,
                        default=Path(os.environ.get('RAM_RESULTS_ROOT', BASE / 'turb-recovery-20260916')))
    parser.add_argument('--previous-results-root', type=Path, action='append',
                        default=[BASE / 'rerun-20260914'])
    args = parser.parse_args()
    if args.submit:
        # Shared lock prevents simultaneous invocations racing each other's queue checks.
        (HERE / 'out').mkdir(exist_ok=True)
        with (HERE / 'out' / '.turb-recovery.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
