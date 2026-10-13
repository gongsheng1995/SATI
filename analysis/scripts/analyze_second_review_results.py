"""Combine real first- and second-round artifacts; never infer missing runs.

Three optimization seeds reuse one fixed mask realization per random control.
Only queue-complete runs with independently reproduced metrics enter summaries.
This is a local CPU analysis, not a training or synthetic testing command.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import analyze_review_results as audit

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
CONTROLS = {
    'original': ('all_true_focal', 'no_prior_focal', 'density_matched_focal',
                 'class_conditioned_focal', 'uncertainty_all_true_focal',
                 'pcgrad_all_true_focal', 'joint_class_mask42_focal'),
    'debiased': ('all_true_focal', 'no_prior_focal', 'joint_class_mask42_focal'),
}
COMPLETE = {'complete', 'complete_reused'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(job, new_round):
    if not new_round:
        protocol = 'debiased' if job['protocol'] == 'debiased' else 'original'
        return protocol, job['condition'], job['seed']
    name = job['name']
    condition = name.rsplit('_seed', 1)[0]
    if condition.startswith('debiased_'):
        return 'debiased', condition[len('debiased_'):], job['seed']
    if condition.startswith('original_'):
        condition = condition[len('original_'):]
    return 'original', condition, job['seed']


def validate_inputs(run, job, new_round, sources):
    config = run['config']
    path = ROOT/config['data_json']
    if path not in sources:
        sources[path] = read(path)
    source = sources[path]
    labels = {r['id']: r['label'] for r in source['test']}
    if len(labels) != len(source['test']) or set(run['ids']) != set(labels):
        raise ValueError('Test membership mismatch: ' + run['name'])
    truth = np.array([config['label2id'][labels[sid]] for sid in run['ids']])
    if not np.array_equal(truth, run['y']) or config['num_classes'] != 30:
        raise ValueError('Test labels or fixed class universe mismatch: ' + run['name'])
    if new_round:
        if any(config.get(k) != v for k, v in job['parameters'].items()):
            raise ValueError('Config differs from saved plan: ' + run['name'])
        if digest(path) != job['data_sha256']:
            raise ValueError('Split provenance mismatch: ' + run['name'])
        if digest(ROOT/config['prior_file']) != job['prior_sha256']:
            raise ValueError('Mask provenance mismatch: ' + run['name'])
    if job.get('method') and not (ROOT/config['output_dir']/'balancing_history.json').is_file():
        raise ValueError('Missing balancing history: ' + run['name'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261007)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError('Bootstrap repetitions must be positive')
    old = ROOT/'results/reruns'
    new = ROOT/'results/second_review'
    old_queue = read(old/'queue_progress.json')
    balance = read(old/'balancing_progress.json')
    new_queue = read(new/'queue_progress.json')
    plan = read(ROOT/'plans/second_review_phase1.json')
    if new_queue['plan_sha256'] != digest(ROOT/'plans/second_review_phase1.json'):
        raise ValueError('Queue plan hash mismatch')
    planned = {j['name']: j for j in plan['jobs']}
    if {j['name'] for j in new_queue['jobs']} != set(planned):
        raise ValueError('Queue membership differs from frozen plan')
    for job in new_queue['jobs']:
        for key in ('seed', 'parameters', 'method', 'prior_sha256', 'data_sha256'):
            if job.get(key) != planned[job['name']].get(key):
                raise ValueError('Queue specification differs: ' + job['name'])
    groups = defaultdict(dict)
    runs, sources, pending = [], {}, []
    rounds = [('previous', old, old_queue['jobs']+balance['jobs'], False),
              ('second', new, new_queue['jobs'], True)]
    for round_name, folder, jobs, is_new in rounds:
        for job in jobs:
            if job['status'] not in COMPLETE:
                pending.append({'round': round_name, 'name': job['name'],
                                'queue_status': job['status'], 'reason': 'not_queue_complete'})
                continue
            run = audit.load_run(folder/job['name'])
            if run is None:
                pending.append({'round': round_name, 'name': job['name'],
                                'queue_status': job['status'], 'reason': 'not_fully_recovered'})
                continue
            validate_inputs(run, job, is_new, sources)
            protocol, condition, seed = identity(job, is_new)
            if seed in groups[protocol, condition]:
                raise ValueError('Duplicate condition/seed across rounds')
            groups[protocol, condition][seed] = run
            runs.append({'round': round_name, 'name': run['name'], 'protocol': protocol,
                         'condition': condition, 'seed': seed,
                         'prediction_sha256': run['prediction_sha256'],
                         'metrics_percent': (100*run['metrics']).tolist(),
                         'epochs_completed': run['epochs_completed']})
    summaries, comparisons = [], []
    for protocol, controls in CONTROLS.items():
        for condition in ('sati_focal',)+controls:
            data = groups[protocol, condition]
            seeds = sorted(data)
            matrix = np.array([data[s]['metrics'] for s in seeds])
            if seeds:
                first = data[seeds[0]]
                for s in seeds:
                    current = data[s]
                    if (first['ids'] != current['ids'] or
                        not np.array_equal(first['y'], current['y']) or
                        any(first['config'][k] != current['config'][k] for k in audit.MATCH_KEYS)):
                        raise ValueError('Unmatched seeds in group ' + condition)
                    if digest(ROOT/first['config']['prior_file']) != digest(ROOT/current['config']['prior_file']):
                        raise ValueError('Fixed mask changed across optimization seeds')
            summaries.append({'protocol': protocol, 'condition': condition, 'seeds': seeds,
                'required_seeds': list(SEEDS), 'status': 'complete' if seeds == list(SEEDS) else 'partial',
                'n_test': len(data[seeds[0]]['ids']) if seeds else None,
                'mean_percent': (100*matrix.mean(0)).tolist() if seeds else None,
                'sample_sd_percent': (100*matrix.std(0, ddof=1)).tolist() if len(seeds)>1 else None,
                'mask_generation_seed': 42 if condition in ('density_matched_focal',
                    'class_conditioned_focal', 'joint_class_mask42_focal') else None,
                'mask_realizations': 1 if condition in ('density_matched_focal',
                    'class_conditioned_focal', 'joint_class_mask42_focal') else None})
        for index, control in enumerate(controls):
            result = audit.paired_comparison(groups[protocol, 'sati_focal'],
                groups[protocol, control], args.repetitions,
                args.bootstrap_seed+index+(100 if protocol=='debiased' else 0), SEEDS)
            result.update(protocol=protocol, left='sati_focal', right=control)
            comparisons.append(result)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete' if len(runs)==36 and not pending else 'partial',
        'previous_completed_and_audited': sum(r['round']=='previous' for r in runs),
        'new_completed_and_audited': sum(r['round']=='second' for r in runs),
        'combined_audited': len(runs), 'combined_planned': 36,
        'new_queue_status': new_queue['status'], 'current_run': new_queue.get('current'),
        'absolute_deadline_utc': new_queue['absolute_deadline'],
        'metric_order': ['ACC', 'WF1', 'MF1_fixed_30_classes'],
        'plan_sha256': digest(ROOT/'plans/second_review_phase1.json'),
        'pending_runs': pending, 'groups': summaries, 'paired_comparisons': comparisons, 'runs': runs,
        'cautions': ['Three optimization seeds are not repeated mask or LLM generation.',
            'Paired bootstrap intervals are pointwise and exploratory with only three seeds; no dialogue clustering or multiple comparison adjustment.',
            'Partial groups have descriptive means only, no full three-seed bootstrap inference.',
            'PCGrad acts only on the nonverbal projection union; uncertainty weighting is an all-sample adaptation.',
            'Same ordinary best-validation-accuracy checkpoint protocol is retained.']}
    path = new/'combined_review_analysis.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'previous_completed_and_audited',
        'new_completed_and_audited', 'combined_audited', 'combined_planned')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
