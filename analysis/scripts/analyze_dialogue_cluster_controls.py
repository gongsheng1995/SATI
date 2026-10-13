"""CPU sensitivity analysis of recovered controls; no training or model loading."""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import analyze_review_results as audit
import analyze_second_review_results as combined

ROOT = combined.ROOT


def main():
    source_path = ROOT / 'results/second_review/combined_review_analysis.json'
    source = combined.read(source_path)
    if source['status'] != 'complete' or source['combined_audited'] != 36:
        raise ValueError('Requires the complete audited 36-run report')
    tsv = ROOT / 'data/retrain_inputs/text/test.tsv'
    with tsv.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    official = {f"dia{r['Dialogue_id']}_utt{r['Utterance_id']}": r for r in rows}
    if len(official) != len(rows):
        raise ValueError('Duplicate official test IDs')
    groups = defaultdict(dict)
    for entry in source['runs']:
        directory = ROOT / ('results/reruns' if entry['round'] == 'previous'
                            else 'results/second_review') / entry['name']
        run = audit.load_run(directory)
        if run is None or run['prediction_sha256'] != entry['prediction_sha256']:
            raise ValueError('Recovered run changed or incomplete')
        for sid, y in zip(run['ids'], run['y']):
            if sid not in official or run['config']['label2id'][official[sid]['Label']] != y:
                raise ValueError('Official ID/label mismatch')
        groups[entry['protocol'], entry['condition']][entry['seed']] = run
    output = []
    memberships = {}
    for index, prior in enumerate(source['paired_comparisons']):
        protocol, control = prior['protocol'], prior['right']
        left, right = groups[protocol, 'sati_focal'], groups[protocol, control]
        seeds = list(combined.SEEDS)
        if sorted(left) != seeds or sorted(right) != seeds:
            raise ValueError('Requires all three paired optimization seeds')
        first = left[seeds[0]]
        for s in seeds:
            for run in (left[s], right[s]):
                if (run['ids'] != first['ids'] or not np.array_equal(run['y'], first['y'])
                        or any(run['config'][k] != first['config'][k] for k in audit.MATCH_KEYS)):
                    raise ValueError('Unmatched data or training protocol')
        dialogue_ids = sorted({official[sid]['Dialogue_id'] for sid in first['ids']})
        lookup = {sid: i for i, sid in enumerate(dialogue_ids)}
        membership = np.array([lookup[official[sid]['Dialogue_id']] for sid in first['ids']])
        sizes = np.bincount(membership, minlength=len(dialogue_ids))
        memberships[protocol] = {'n_test': len(membership), 'n_dialogues': len(sizes),
            'min_cluster_size': int(sizes.min()), 'max_cluster_size': int(sizes.max()),
            'dialogue_ids': dialogue_ids, 'cluster_sizes': sizes.tolist()}
        random_seed = 20261008 + index
        rng = np.random.default_rng(random_seed)
        samples = np.empty((5000, 3))
        for repetition in range(len(samples)):
            counts = np.bincount(rng.integers(len(sizes), size=len(sizes)), minlength=len(sizes))
            weights = counts[membership]
            chosen = rng.integers(3, size=3)
            samples[repetition] = np.mean([
                audit.metrics_from_codes(left[seeds[j]]['codes'], weights, 30)
                - audit.metrics_from_codes(right[seeds[j]]['codes'], weights, 30)
                for j in chosen], axis=0)
        delta = 100 * np.mean([left[s]['metrics'] - right[s]['metrics'] for s in seeds], axis=0)
        if not np.allclose(delta, prior['mean_delta_percentage_points'], atol=1e-9):
            raise ValueError('Point estimate differs from previous analysis')
        output.append({'protocol': protocol, 'left': 'sati_focal', 'right': control,
            'seeds': seeds, 'mean_delta_percentage_points': delta.tolist(),
            'repetitions': len(samples), 'random_seed': random_seed,
            'lower_95_percentage_points': (100*np.quantile(samples, .025, axis=0)).tolist(),
            'upper_95_percentage_points': (100*np.quantile(samples, .975, axis=0)).tolist()})
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
        'source_report_sha256': combined.digest(source_path), 'official_test_tsv_sha256': combined.digest(tsv),
        'analysis_source_sha256': combined.digest(Path(__file__)),
        'metric_order': source['metric_order'], 'memberships': memberships, 'comparisons': output,
        'method': 'Paired percentile bootstrap: sample test dialogue clusters with replacement, retain all utterances in each selected cluster; independently resample three paired optimization seeds, average seed-wise utterance-weighted metrics. Shared clusters and seed draws across both methods; fixed 30-class macro-F1.',
        'cautions': ['Exploratory pointwise intervals with only three optimization seeds; no multiple-comparison adjustment.',
                     'Dialogue clustering addresses within-dialogue dependence only, not shared speakers or sources across dialogues.',
                     'One mask realization per random condition remains unchanged; this is not mask-generation replication.',
                     'Original example-bootstrap files are preserved. No model loading, training or new predictions.']}
    path = ROOT / 'results/second_review/dialogue_cluster_control_analysis.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'clusters': {p: m['n_dialogues'] for p, m in memberships.items()},
                      'comparisons': output}, ensure_ascii=False))


if __name__ == '__main__':
    main()
