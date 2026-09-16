import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_turb_remaining as recovery


class TurbulenceRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = self.root / 'data' / 'forced_turb'
        data.mkdir(parents=True)
        for name in ('data.mat', 'data_ood.mat'):
            (data / name).touch()
        self.environment = patch.dict(os.environ, RAM_DATA_ROOT=str(data.parent))
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.args = type('Args', (), dict(results_root=self.root / 'results', model='all',
                                        phase='all', seed=None, ntrain=None))()

    def plan(self):
        with contextlib.redirect_stderr(io.StringIO()):
            return recovery.plan(self.args)

    def select(self, jobs, done=set(), active=set()):
        with contextlib.redirect_stderr(io.StringIO()):
            return recovery.select(jobs, done, active, [self.args.results_root])

    def test_snapshot_skips_exactly_completed_configurations(self):
        jobs = self.plan()
        done = recovery.known_completed(jobs, [])
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len(done), 23)
        selected = self.select(jobs, done)
        self.assertEqual(len(selected), 37)
        self.assertEqual(sum(j['model'] == 'geo' for j in selected), 21)
        self.assertEqual(sum(j['model'] == 'trans' for j in selected), 16)
        self.assertIn(('trans', 1, 5000, 0.), done)
        self.assertNotIn(('trans', 2, 5000, 0.), done)

    def test_changed_hyperparameters_do_not_match_snapshot(self):
        jobs = self.plan()
        for job in jobs:
            job['config']['epochs'] = '1000'
        self.assertFalse(recovery.known_completed(jobs, []))

    def test_hours_have_measured_headroom(self):
        self.assertEqual([recovery.walltime('geo', n) for n in (5000, 7000, 10000)], [24, 32, 46])
        self.assertEqual([recovery.walltime('trans', n) for n in (5000, 7000, 10000)], [6, 8, 10])

    def test_pending_and_running_names_with_old_and_new_prefixes(self):
        result = type('Result', (), {'stdout': 'rerun_geo_forced_turb_s1_n10000_l0.001\n'
                                    'turbfix_trans_forced_turb_s2_n7000_l0\n'})()
        with patch.object(recovery.shutil, 'which', return_value='/bin/squeue'), \
                patch.object(recovery.subprocess, 'run', return_value=result):
            active = recovery.queue_keys(True)
        self.assertEqual(active, {('geo', 1, 10000, .001), ('trans', 2, 7000, 0.)})
        self.assertEqual(len(self.select(self.plan(), active=active)), 58)

    def test_no_submission_without_queue_access(self):
        with patch.object(recovery.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'refusing submission'):
                recovery.queue_keys(True)

    def test_missing_ood_file_excludes_baselines_not_div(self):
        # Point to a separate train-only directory; leave existing fixtures untouched.
        train_only = self.root / 'train_only' / 'forced_turb'
        train_only.mkdir(parents=True)
        (train_only / 'data.mat').touch()
        with patch.dict(os.environ, RAM_DATA_ROOT=str(train_only.parent)):
            jobs = self.plan()
        self.assertEqual(len(jobs), 24)
        self.assertTrue(all(j['phase'] == 'div' for j in jobs))

    def test_new_wandb_completion_skips_another_seed(self):
        job = self.plan()[0]
        folder = self.root / 'wandb' / 'run-new' / 'files'
        folder.mkdir(parents=True)
        (folder / 'wandb-metadata.json').write_text(json.dumps(
            {'program': 'ramansh_3d.py', 'args': job['command']}))
        metrics = {'test_loss': .01, 'test_div/max_abs_interior': .2,
                   'test_div/median_abs_interior': .1}
        (folder / 'wandb-summary.json').write_text(json.dumps(metrics))
        done = recovery.known_completed([job], [self.root / 'wandb'])
        self.assertEqual(done, {('geo', 1, 10000, .001)})
        self.assertFalse(recovery.complete(metrics, 0.))

    def test_checkpoint_resumes_once_and_final_metrics_skip(self):
        job = self.plan()[0]
        path = recovery.checkpoint_paths(job, [self.args.results_root])[0]
        path.parent.mkdir(parents=True)
        path.touch()
        self.select([job])
        self.select([job])
        self.assertEqual(sum(t.startswith('--resume=') for t in job['command']), 1)
        path.with_suffix('.metrics.json').write_text(json.dumps({
            'config': job['config'], 'metrics': {'test_loss': .01,
            'test_div/max_abs_interior': .2, 'test_div/median_abs_interior': .1}}))
        self.assertFalse(self.select([job]))

    def test_script_resources_and_training_objective(self):
        for job in self.plan():
            script = recovery.batch_script(job)
            self.assertIn('#SBATCH --partition=gpuA100x4', script)
            self.assertIn('#SBATCH --account=bgcs-delta-gpu', script)
            self.assertEqual(job['config']['epochs'], '500')
            self.assertEqual(job['config']['npoints'], '7000')
            self.assertEqual('--div-loss' in job['command'], job['phase'] == 'div')
            self.assertEqual('--require-ood' in job['command'], job['phase'] != 'div')


if __name__ == '__main__':
    unittest.main()
