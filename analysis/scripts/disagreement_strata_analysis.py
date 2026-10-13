"""Disagreement-strata analysis: does SATI's test-time edge concentrate on prior-selected samples?

For each protocol (original, debiased) and seed, pair SATI vs all-sample predictions by test ID,
join the frozen prior archive's TEST judgments, and stratify:
  stratum P = prior assigns at least one nonverbal mask (audio or video = 1)
  stratum N = neither mask selected (all-sample supervises them; SATI does not)
Report per-stratum accuracy for both methods, flip counts, and paired differences.
Read-only; writes paper_work/second_review/disagreement_strata_results.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('SATI 9')
WORK = Path('paper_work/second_review')

prior = {}
for line in (ROOT / 'data/llm_prior.jsonl').open(encoding='utf-8'):
    if line.strip():
        d = json.loads(line)
        if d['split'] == 'test':
            prior[d['id']] = (bool(d['audio_necessary']), bool(d['video_necessary']))

def load_preds(run):
    rows = [json.loads(x) for x in (ROOT / f'results/reruns/{run}/test_predictions.jsonl').read_text().splitlines() if x.strip()]
    return {r['id']: r for r in rows}

out = {'protocols': {}}
for proto, prefix in (('original', ''), ('debiased', 'debiased_')):
    proto_res = {'seeds': [], 'strata': {}}
    acc = {'sati_P': [], 'all_P': [], 'sati_N': [], 'all_N': []}
    for seed in (42, 43, 44):
        sati = load_preds(f'{prefix}sati_focal_seed{seed}')
        alls = load_preds(f'{prefix}all_true_focal_seed{seed}')
        ids = sorted(set(sati) & set(alls))
        assert len(ids) == len(sati) == len(alls)
        strata = {'P': {'sati': [], 'all': []}, 'N': {'sati': [], 'all': []}}
        flips = {'P': [0, 0], 'N': [0, 0]}  # [sati_correct_all_wrong, sati_wrong_all_correct]
        for i in ids:
            a, v = prior[i]
            s = 'P' if (a or v) else 'N'
            sc = sati[i]['pred_id'] == sati[i]['target_id']
            ac = alls[i]['pred_id'] == alls[i]['target_id']
            strata[s]['sati'].append(sc)
            strata[s]['all'].append(ac)
            if sc and not ac:
                flips[s][0] += 1
            elif ac and not sc:
                flips[s][1] += 1
        seed_res = {'seed': seed}
        for s in ('P', 'N'):
            n = len(strata[s]['sati'])
            seed_res[s] = {
                'n': n,
                'sati_acc': float(np.mean(strata[s]['sati'])) if n else None,
                'all_acc': float(np.mean(strata[s]['all'])) if n else None,
                'sati_only_correct': flips[s][0],
                'all_only_correct': flips[s][1],
            }
            if n:
                acc[f'sati_{s}'].append(seed_res[s]['sati_acc'])
                acc[f'all_{s}'].append(seed_res[s]['all_acc'])
        proto_res['seeds'].append(seed_res)
    proto_res['summary'] = {
        s: {
            'n': proto_res['seeds'][0][s]['n'],
            'sati_acc_mean': float(np.mean(acc[f'sati_{s}'])),
            'all_acc_mean': float(np.mean(acc[f'all_{s}'])),
            'diff_pp': 100 * (float(np.mean(acc[f'sati_{s}'])) - float(np.mean(acc[f'all_{s}']))),
        } for s in ('P', 'N')
    }
    out['protocols'][proto] = proto_res

out['method'] = ('Test samples stratified by frozen prior test judgments: P = audio or video mask positive '
                 '(selected for auxiliary supervision), N = neither selected (all-sample supervises, SATI does not). '
                 'Accuracy is over per-stratum samples; three seeds averaged; flips are per-seed totals summed.')
(WORK / 'disagreement_strata_results.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
for proto in ('original', 'debiased'):
    s = out['protocols'][proto]['summary']
    print(f"{proto}: P n={s['P']['n']} SATI {100*s['P']['sati_acc_mean']:.2f} vs ALL {100*s['P']['all_acc_mean']:.2f} (diff {s['P']['diff_pp']:+.2f}pp) | "
          f"N n={s['N']['n']} SATI {100*s['N']['sati_acc_mean']:.2f} vs ALL {100*s['N']['all_acc_mean']:.2f} (diff {s['N']['diff_pp']:+.2f}pp)")
